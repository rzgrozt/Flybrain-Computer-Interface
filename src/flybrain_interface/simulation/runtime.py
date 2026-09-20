"""Deterministic CPU LIF runtime with sparse delayed synaptic events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, isclose
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from flybrain_interface.contracts import NeuralReadout
from flybrain_interface.sensory.drive import DeterministicProjectedDriveInput
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.kernels import (
    advance_state_numba,
    advance_state_numba_zero_subnormal,
)
from flybrain_interface.simulation.trace import (
    ChunkRecording,
    ChunkResult,
    SimulationTrace,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
RuntimeBackend = Literal["numpy", "numba"]
SubnormalDrivePolicy = Literal["preserve", "zero"]


class EventConnectivity(Protocol):
    @property
    def neuron_count(self) -> int: ...

    def accumulate_spikes(
        self,
        spiking_indices: npt.ArrayLike,
        destination: FloatArray,
        *,
        amplitudes: npt.ArrayLike | None = None,
    ) -> int: ...


@dataclass(slots=True)
class SparseLIFSimulator:
    """Stateful Shiu-style LIF engine using a sparse source-event delay ring."""

    connectivity: EventConnectivity
    populations: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    clamped_indices: tuple[int, ...] = ()
    config: ShiuLIFConfig = field(default_factory=ShiuLIFConfig)
    backend: RuntimeBackend = "numba"
    subnormal_drive_policy: SubnormalDrivePolicy = "preserve"
    tonic_bias_overrides_mv: Mapping[int, float] = field(default_factory=dict)
    threshold_overrides_mv: Mapping[int, float] = field(default_factory=dict)
    graded_relay_indices: tuple[int, ...] = ()
    graded_relay_gain: float = 0.0
    graded_relay_activation_scale_mv: float = 7.0
    voltage_mv: FloatArray = field(init=False, repr=False)
    synaptic_drive_mv: FloatArray = field(init=False, repr=False)
    refractory_steps_left: IntArray = field(init=False, repr=False)
    _delay_ring: list[IntArray] = field(init=False, repr=False)
    _membrane_decay: float = field(init=False, repr=False)
    _synapse_decay: float = field(init=False, repr=False)
    _drive_coupling: float = field(init=False, repr=False)
    _minimum_preserved_drive_mv: float = field(init=False, repr=False)
    _spike_buffer: IntArray = field(init=False, repr=False)
    _refractory_index_buffer: IntArray = field(init=False, repr=False)
    _refractory_drive_buffer: FloatArray = field(init=False, repr=False)
    _clamped_index_array: IntArray = field(init=False, repr=False)
    _tonic_bias_mv: FloatArray = field(init=False, repr=False)
    _threshold_mv: FloatArray = field(init=False, repr=False)
    _graded_relay_index_array: IntArray = field(init=False, repr=False)
    _step_index: int = field(init=False, default=0, repr=False)
    visited_edges: int = field(init=False, default=0)
    projected_drive_target_updates: int = field(init=False, default=0)
    state_update_seconds: float = field(init=False, default=0.0)
    synaptic_propagation_seconds: float = field(init=False, default=0.0)
    recording_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.connectivity.neuron_count <= 0:
            raise ValueError("connectivity must contain neurons")
        if self.backend not in ("numpy", "numba"):
            raise ValueError(f"unsupported runtime backend: {self.backend}")
        if self.subnormal_drive_policy not in ("preserve", "zero"):
            raise ValueError(
                f"unsupported subnormal drive policy: {self.subnormal_drive_policy}"
            )
        self._validate_populations()
        self._clamped_index_array = self._validate_clamped_indices(
            self.clamped_indices
        )
        self._graded_relay_index_array = self._validate_external_indices(
            self.graded_relay_indices
        )
        if not np.isfinite(self.graded_relay_gain) or self.graded_relay_gain < 0.0:
            raise ValueError("graded_relay_gain must be finite and non-negative")
        if (
            not np.isfinite(self.graded_relay_activation_scale_mv)
            or self.graded_relay_activation_scale_mv <= 0.0
        ):
            raise ValueError(
                "graded_relay_activation_scale_mv must be finite and positive"
            )
        cfg = self.config
        self._tonic_bias_mv, self._threshold_mv = self._build_excitability_arrays()
        self._membrane_decay = exp(-cfg.dt_ms / cfg.membrane_tau_ms)
        self._synapse_decay = exp(-cfg.dt_ms / cfg.synapse_tau_ms)
        self._drive_coupling = (
            cfg.synapse_tau_ms
            / (cfg.synapse_tau_ms - cfg.membrane_tau_ms)
            * (self._synapse_decay - self._membrane_decay)
        )
        self._minimum_preserved_drive_mv = (
            float(np.finfo(np.float64).tiny / self._synapse_decay)
            if self.subnormal_drive_policy == "zero"
            else 0.0
        )
        self.reset()

    def reset(self) -> None:
        count = self.connectivity.neuron_count
        self.voltage_mv = np.full(count, self.config.resting_mv, dtype=np.float64)
        self.synaptic_drive_mv = np.zeros(count, dtype=np.float64)
        self.refractory_steps_left = np.zeros(count, dtype=np.int64)
        self._spike_buffer = np.empty(count, dtype=np.int64)
        self._refractory_index_buffer = np.empty(count, dtype=np.int64)
        self._refractory_drive_buffer = np.empty(count, dtype=np.float64)
        ring_size = max(1, self.config.delay_steps + 1)
        self._delay_ring = [np.empty(0, dtype=np.int64) for _ in range(ring_size)]
        self._step_index = 0
        self.visited_edges = 0
        self.projected_drive_target_updates = 0
        self.state_update_seconds = 0.0
        self.synaptic_propagation_seconds = 0.0
        self.recording_seconds = 0.0

    def prepare(self) -> None:
        """Compile/load the selected backend without advancing persistent state."""

        if self.backend == "numba":
            self._advance_state_numba()
            self.reset()

    def _advance_state_numba(self) -> tuple[int, int]:
        arguments = (
            self.voltage_mv,
            self.synaptic_drive_mv,
            self.refractory_steps_left,
            self._spike_buffer,
            self._refractory_index_buffer,
            self._refractory_drive_buffer,
            self.config.resting_mv,
            self._threshold_mv,
            self._tonic_bias_mv,
            self._membrane_decay,
            self._synapse_decay,
            self._drive_coupling,
        )
        if self.subnormal_drive_policy == "zero":
            return advance_state_numba_zero_subnormal(
                *arguments, self._minimum_preserved_drive_mv
            )
        return advance_state_numba(*arguments)

    def run(
        self,
        stimulus: DeterministicSpikeInput,
        *,
        duration_s: float,
        watched_indices: tuple[int, ...] = (),
    ) -> SimulationTrace:
        """Reset, execute a fixed-duration run, and return comparable traces."""

        step_count = _duration_steps(duration_s, self.config.dt_ms)
        self.reset()
        chunk = self.advance_chunk(
            stimulus,
            duration_s=duration_s,
            recording=ChunkRecording(
                watched_indices=watched_indices,
                include_neuron_counts=True,
                max_spike_events=step_count * self.connectivity.neuron_count,
            ),
        )
        spike_times: list[list[float]] = [
            [] for _ in range(self.connectivity.neuron_count)
        ]
        for neuron, time_s in zip(
            chunk.spike_neuron_indices, chunk.spike_times_s, strict=True
        ):
            spike_times[int(neuron)].append(float(time_s))
        immutable_times = tuple(tuple(times) for times in spike_times)
        if chunk.neuron_spike_counts is None or chunk.dropped_spike_events:
            raise RuntimeError(
                "compatibility run failed to retain complete spike history"
            )
        readout = NeuralReadout(
            duration_s=duration_s,
            neuron_spike_counts=tuple(
                int(value) for value in chunk.neuron_spike_counts
            ),
            population_rates_hz=chunk.population_rates_hz,
            spike_times_s=immutable_times,
            population_voltage_delta_mv=chunk.population_voltage_delta_mv,
            population_synaptic_drive_mv=chunk.population_synaptic_drive_mv,
        )
        return SimulationTrace(
            readout=readout,
            watched_indices=watched_indices,
            sample_times_s=chunk.sample_times_s,
            voltage_mv=chunk.voltage_mv,
            synaptic_drive_mv=chunk.synaptic_drive_mv,
        )

    @property
    def current_step(self) -> int:
        """Current absolute neural-grid step since the last reset."""

        return self._step_index

    @property
    def current_time_s(self) -> float:
        """Current absolute simulated time since the last reset."""

        return self._step_index * self.config.dt_ms / 1000.0

    @property
    def pending_delayed_events(self) -> int:
        """Number of emitted spikes awaiting delivery in the delay ring."""

        return sum(int(events.size) for events in self._delay_ring)

    def advance_step(
        self,
        input_indices: npt.ArrayLike = (),
        *,
        amplitude_mv: float = 8.0,
    ) -> IntArray:
        """Advance one grid point without resetting and return emitted neuron IDs."""

        indices = self._validate_external_indices(input_indices)
        if not np.isfinite(amplitude_mv):
            raise ValueError("stimulus amplitude must be finite")
        return self._advance(indices if indices.size else None, amplitude_mv)

    def advance_chunk(
        self,
        stimulus: DeterministicSpikeInput | None = None,
        *,
        duration_s: float,
        recording: ChunkRecording = ChunkRecording(),
        projected_drive: DeterministicProjectedDriveInput | None = None,
    ) -> ChunkResult:
        """Advance persistent state; all supplied inputs are relative to this chunk."""

        if stimulus is None:
            stimulus = DeterministicSpikeInput(neuron_indices=(), times_s=())
        step_count = _duration_steps(duration_s, self.config.dt_ms)
        watch = self._validated_watch(recording.watched_indices)
        input_schedule = self._input_schedule(stimulus, step_count)
        self._validate_projected_drive(projected_drive, step_count)
        start_step = self._step_index
        sample_times = (start_step + np.arange(step_count, dtype=np.float64)) * (
            self.config.dt_ms / 1000.0
        )
        voltage_trace = np.empty((step_count, watch.size), dtype=np.float64)
        drive_trace = np.empty((step_count, watch.size), dtype=np.float64)
        counts = (
            np.zeros(self.connectivity.neuron_count, dtype=np.int64)
            if recording.include_neuron_counts or self.populations
            else None
        )
        event_neurons: list[int] = []
        event_times: list[float] = []
        total_spikes = 0

        for local_step in range(step_count):
            external_indices = input_schedule.get(local_step)
            spikes = self._advance(
                external_indices,
                stimulus.amplitude_mv,
                projected_drive=projected_drive,
                projected_drive_step=local_step,
            )
            recording_started = perf_counter()
            spike_count = int(spikes.size)
            total_spikes += spike_count
            if spike_count and counts is not None:
                np.add.at(counts, spikes, 1)
            if spike_count:
                remaining = recording.max_spike_events - len(event_neurons)
                if remaining > 0:
                    retained = spikes[:remaining]
                    event_neurons.extend(int(neuron) for neuron in retained)
                    event_times.extend(
                        [float(sample_times[local_step])] * int(retained.size)
                    )
            if watch.size:
                voltage_trace[local_step] = self.voltage_mv[watch]
                drive_trace[local_step] = self.synaptic_drive_mv[watch]
            self.recording_seconds += perf_counter() - recording_started

        rates = (
            {
                name: float(np.sum(counts[np.asarray(indices, dtype=np.int64)]))
                / (len(indices) * duration_s)
                for name, indices in self.populations.items()
            }
            if counts is not None
            else {}
        )
        population_voltage_delta_mv = {
            name: float(
                np.mean(
                    self.voltage_mv[np.asarray(indices, dtype=np.int64)]
                    - self.config.resting_mv
                )
            )
            for name, indices in self.populations.items()
        }
        population_synaptic_drive_mv = {
            name: float(
                np.mean(
                    self.synaptic_drive_mv[np.asarray(indices, dtype=np.int64)]
                )
            )
            for name, indices in self.populations.items()
        }
        recorded_count = len(event_neurons)
        return ChunkResult(
            start_step=start_step,
            end_step=self._step_index,
            dt_ms=self.config.dt_ms,
            total_spikes=total_spikes,
            population_rates_hz=rates,
            neuron_spike_counts=(counts if recording.include_neuron_counts else None),
            spike_neuron_indices=np.asarray(event_neurons, dtype=np.int64),
            spike_times_s=np.asarray(event_times, dtype=np.float64),
            dropped_spike_events=total_spikes - recorded_count,
            watched_indices=recording.watched_indices,
            sample_times_s=sample_times,
            voltage_mv=voltage_trace,
            synaptic_drive_mv=drive_trace,
            population_voltage_delta_mv=population_voltage_delta_mv,
            population_synaptic_drive_mv=population_synaptic_drive_mv,
        )

    def _advance(
        self,
        external_indices: IntArray | None,
        external_amplitude_mv: float,
        *,
        projected_drive: DeterministicProjectedDriveInput | None = None,
        projected_drive_step: int = 0,
    ) -> IntArray:
        cfg = self.config
        state_started = perf_counter()
        if self.backend == "numba":
            spike_count, refractory_count = self._advance_state_numba()
        else:
            spike_count, refractory_count = self._advance_state_numpy()
        self.state_update_seconds += perf_counter() - state_started
        spikes = self._spike_buffer[:spike_count]
        refractory_indices = self._refractory_index_buffer[:refractory_count]
        refractory_drive = self._refractory_drive_buffer[:refractory_count]
        if self._clamped_index_array.size:
            self._apply_clamp()
            if spikes.size:
                spikes = spikes[
                    ~np.isin(
                        spikes,
                        self._clamped_index_array,
                        assume_unique=True,
                    )
                ]

        propagation_started = perf_counter()
        due_slot = self._step_index % len(self._delay_ring)
        due_sources = self._delay_ring[due_slot]
        self._delay_ring[due_slot] = np.empty(0, dtype=np.int64)
        if due_sources.size:
            self.visited_edges += self.connectivity.accumulate_spikes(
                due_sources,
                self.synaptic_drive_mv,
                amplitudes=cfg.synapse_scale_mv,
            )
        if projected_drive is not None:
            for channel in projected_drive.channels:
                amplitude = float(channel.amplitudes_mv[projected_drive_step])
                if amplitude == 0.0 or channel.target_indices.size == 0:
                    continue
                self.synaptic_drive_mv[channel.target_indices] += (
                    channel.signed_contact_weights * amplitude
                )
                self.projected_drive_target_updates += int(channel.target_indices.size)
        graded_wrote = False
        if self._graded_relay_index_array.size and self.graded_relay_gain > 0.0:
            relay_indices = self._graded_relay_index_array
            voltage_delta = self.voltage_mv[relay_indices] - cfg.resting_mv
            active = voltage_delta > 0.0
            if spikes.size:
                active &= ~np.isin(relay_indices, spikes, assume_unique=True)
            if np.any(active):
                active_sources = relay_indices[active]
                release_fraction = np.clip(
                    voltage_delta[active] / self.graded_relay_activation_scale_mv,
                    0.0,
                    1.0,
                )
                amplitudes = (
                    cfg.synapse_scale_mv
                    * self.graded_relay_gain
                    * release_fraction
                )
                self.visited_edges += self.connectivity.accumulate_spikes(
                    active_sources,
                    self.synaptic_drive_mv,
                    amplitudes=amplitudes,
                )
                graded_wrote = True
        if due_sources.size or projected_drive is not None or graded_wrote:
            # Refractory state suppresses all synaptic/drive writes to g.
            self.synaptic_drive_mv[refractory_indices] = refractory_drive
        self.synaptic_propagation_seconds += perf_counter() - propagation_started
        if external_indices is not None:
            receptive = external_indices[
                ~np.isin(external_indices, refractory_indices, assume_unique=True)
            ]
            self.voltage_mv[receptive] += external_amplitude_mv

        if spikes.size:
            emitted_spikes = spikes.copy()
            if cfg.delay_steps == 0:
                propagation_started = perf_counter()
                self.visited_edges += self.connectivity.accumulate_spikes(
                    spikes,
                    self.synaptic_drive_mv,
                    amplitudes=cfg.synapse_scale_mv,
                )
                self.synaptic_drive_mv[refractory_indices] = refractory_drive
                self.synaptic_propagation_seconds += (
                    perf_counter() - propagation_started
                )
            else:
                delivery_slot = (self._step_index + cfg.delay_steps) % len(
                    self._delay_ring
                )
                self._delay_ring[delivery_slot] = emitted_spikes
            self.voltage_mv[spikes] = cfg.reset_mv
            self.synaptic_drive_mv[spikes] = 0.0
            self.refractory_steps_left[spikes] = max(0, cfg.refractory_steps - 1)

        self._apply_clamp()
        self._step_index += 1
        return spikes.copy() if not spikes.size else emitted_spikes

    def _advance_state_numpy(self) -> tuple[int, int]:
        cfg = self.config
        active = self.refractory_steps_left == 0
        refractory_indices = np.flatnonzero(~active)
        refractory_voltage = self.voltage_mv[refractory_indices].copy()
        refractory_drive = self.synaptic_drive_mv[refractory_indices].copy()
        if self._minimum_preserved_drive_mv > 0.0:
            small_drive = (self.synaptic_drive_mv != 0.0) & (
                np.abs(self.synaptic_drive_mv) < self._minimum_preserved_drive_mv
            )
            self.synaptic_drive_mv[small_drive] = 0.0
        equilibrium_mv = cfg.resting_mv + self._tonic_bias_mv
        self.voltage_mv -= equilibrium_mv
        self.voltage_mv *= self._membrane_decay
        self.voltage_mv += equilibrium_mv
        self.voltage_mv += self.synaptic_drive_mv * self._drive_coupling
        self.synaptic_drive_mv *= self._synapse_decay
        self.voltage_mv[refractory_indices] = refractory_voltage
        self.synaptic_drive_mv[refractory_indices] = refractory_drive
        self.refractory_steps_left[~active] -= 1

        spikes = np.flatnonzero(active & (self.voltage_mv > self._threshold_mv)).astype(
            np.int64, copy=False
        )
        self._spike_buffer[: spikes.size] = spikes
        self._refractory_index_buffer[: refractory_indices.size] = refractory_indices
        self._refractory_drive_buffer[: refractory_indices.size] = refractory_drive
        return int(spikes.size), int(refractory_indices.size)

    def _input_schedule(
        self, stimulus: DeterministicSpikeInput, step_count: int
    ) -> dict[int, IntArray]:
        if not np.isfinite(stimulus.amplitude_mv):
            raise ValueError("stimulus amplitude must be finite")
        grouped: dict[int, list[int]] = {}
        for neuron, time_s in zip(
            stimulus.neuron_indices, stimulus.times_s, strict=True
        ):
            if neuron < 0 or neuron >= self.connectivity.neuron_count:
                raise ValueError("stimulus neuron index outside network")
            exact_step = time_s * 1000.0 / self.config.dt_ms
            step = round(exact_step)
            if not isclose(exact_step, step, abs_tol=1e-9):
                raise ValueError("stimulus times must align to dt_ms")
            if step < 0 or step >= step_count:
                raise ValueError("stimulus spike time must be inside the run duration")
            grouped.setdefault(step, []).append(neuron)
        schedule: dict[int, IntArray] = {}
        for step, neurons in grouped.items():
            values = np.asarray(neurons, dtype=np.int64)
            if np.unique(values).size != values.size:
                raise ValueError("a neuron cannot receive duplicate input at one time")
            schedule[step] = values
        return schedule


    def _apply_clamp(self) -> None:
        if not self._clamped_index_array.size:
            return
        self.voltage_mv[self._clamped_index_array] = self.config.resting_mv
        self.synaptic_drive_mv[self._clamped_index_array] = 0.0
        self.refractory_steps_left[self._clamped_index_array] = 0

    def _validate_clamped_indices(self, indices: tuple[int, ...]) -> IntArray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or np.unique(values).size != values.size:
            raise ValueError("clamped_indices must be one-dimensional and unique")
        if values.size and (
            values.min() < 0 or values.max() >= self.connectivity.neuron_count
        ):
            raise ValueError("clamped neuron index outside network")
        return values

    def _build_excitability_arrays(self) -> tuple[FloatArray, FloatArray]:
        count = self.connectivity.neuron_count
        tonic = np.full(count, self.config.tonic_bias_mv, dtype=np.float64)
        threshold = np.full(count, self.config.threshold_mv, dtype=np.float64)
        for index, value in self.tonic_bias_overrides_mv.items():
            neuron = int(index)
            bias = float(value)
            if neuron < 0 or neuron >= count:
                raise ValueError("tonic bias override index outside network")
            if not np.isfinite(bias):
                raise ValueError("tonic bias override must be finite")
            tonic[neuron] = bias
        for index, value in self.threshold_overrides_mv.items():
            neuron = int(index)
            threshold_value = float(value)
            if neuron < 0 or neuron >= count:
                raise ValueError("threshold override index outside network")
            if not np.isfinite(threshold_value):
                raise ValueError("threshold override must be finite")
            threshold[neuron] = threshold_value
        if np.any(threshold <= self.config.resting_mv):
            raise ValueError("threshold override must exceed resting_mv")
        if np.any(self.config.resting_mv + tonic >= threshold):
            raise ValueError(
                "per-neuron tonic bias must keep zero-input equilibrium below threshold"
            )
        return tonic, threshold

    def _validate_projected_drive(
        self,
        projected_drive: DeterministicProjectedDriveInput | None,
        step_count: int,
    ) -> None:
        if projected_drive is None:
            return
        if not isclose(projected_drive.dt_ms, self.config.dt_ms, abs_tol=1e-12):
            raise ValueError("projected drive dt_ms must match simulator dt_ms")
        if projected_drive.step_count != step_count:
            raise ValueError("projected drive trace length must match chunk duration")
        for channel in projected_drive.channels:
            targets = channel.target_indices
            if targets.size and (
                targets.min() < 0 or targets.max() >= self.connectivity.neuron_count
            ):
                raise ValueError("projected drive target index outside network")

    def _validate_external_indices(self, input_indices: npt.ArrayLike) -> IntArray:
        indices = np.asarray(input_indices, dtype=np.int64)
        if indices.ndim != 1 or np.unique(indices).size != indices.size:
            raise ValueError("input_indices must be one-dimensional and unique")
        if indices.size and (
            indices.min() < 0 or indices.max() >= self.connectivity.neuron_count
        ):
            raise ValueError("stimulus neuron index outside network")
        return indices

    def _validated_watch(self, watched_indices: tuple[int, ...]) -> IntArray:
        watch = np.asarray(watched_indices, dtype=np.int64)
        if watch.ndim != 1 or np.unique(watch).size != watch.size:
            raise ValueError("watched_indices must be one-dimensional and unique")
        if watch.size and (
            watch.min() < 0 or watch.max() >= self.connectivity.neuron_count
        ):
            raise ValueError("watched neuron index outside network")
        return watch

    def _validate_populations(self) -> None:
        for name, indices in self.populations.items():
            if not name or not indices:
                raise ValueError("population names and memberships cannot be empty")
            if any(
                index < 0 or index >= self.connectivity.neuron_count
                for index in indices
            ):
                raise ValueError("population index outside network")


def _duration_steps(duration_s: float, dt_ms: float) -> int:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    exact_steps = duration_s * 1000.0 / dt_ms
    steps = round(exact_steps)
    if not isclose(exact_steps, steps, abs_tol=1e-9):
        raise ValueError("duration_s must be an integer multiple of dt_ms")
    return int(steps)
