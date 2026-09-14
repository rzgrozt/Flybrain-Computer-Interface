"""Deterministic CPU LIF runtime with sparse delayed synaptic events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, isclose
from typing import Protocol

import numpy as np
import numpy.typing as npt

from flybrain_interface.contracts import NeuralReadout
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.trace import SimulationTrace

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


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
    config: ShiuLIFConfig = field(default_factory=ShiuLIFConfig)
    voltage_mv: FloatArray = field(init=False, repr=False)
    synaptic_drive_mv: FloatArray = field(init=False, repr=False)
    refractory_steps_left: IntArray = field(init=False, repr=False)
    _delay_ring: list[IntArray] = field(init=False, repr=False)
    _membrane_decay: float = field(init=False, repr=False)
    _synapse_decay: float = field(init=False, repr=False)
    _drive_coupling: float = field(init=False, repr=False)
    _step_index: int = field(init=False, default=0, repr=False)
    visited_edges: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.connectivity.neuron_count <= 0:
            raise ValueError("connectivity must contain neurons")
        self._validate_populations()
        cfg = self.config
        self._membrane_decay = exp(-cfg.dt_ms / cfg.membrane_tau_ms)
        self._synapse_decay = exp(-cfg.dt_ms / cfg.synapse_tau_ms)
        self._drive_coupling = (
            cfg.synapse_tau_ms
            / (cfg.synapse_tau_ms - cfg.membrane_tau_ms)
            * (self._synapse_decay - self._membrane_decay)
        )
        self.reset()

    def reset(self) -> None:
        count = self.connectivity.neuron_count
        self.voltage_mv = np.full(count, self.config.resting_mv, dtype=np.float64)
        self.synaptic_drive_mv = np.zeros(count, dtype=np.float64)
        self.refractory_steps_left = np.zeros(count, dtype=np.int64)
        ring_size = max(1, self.config.delay_steps + 1)
        self._delay_ring = [np.empty(0, dtype=np.int64) for _ in range(ring_size)]
        self._step_index = 0
        self.visited_edges = 0

    def run(
        self,
        stimulus: DeterministicSpikeInput,
        *,
        duration_s: float,
        watched_indices: tuple[int, ...] = (),
    ) -> SimulationTrace:
        """Reset, execute a fixed-duration run, and return comparable traces."""

        step_count = _duration_steps(duration_s, self.config.dt_ms)
        watch = np.asarray(watched_indices, dtype=np.int64)
        if watch.ndim != 1 or np.unique(watch).size != watch.size:
            raise ValueError("watched_indices must be one-dimensional and unique")
        if watch.size and (
            watch.min() < 0 or watch.max() >= self.connectivity.neuron_count
        ):
            raise ValueError("watched neuron index outside network")
        input_schedule = self._input_schedule(stimulus, step_count)

        self.reset()
        sample_times = np.arange(step_count, dtype=np.float64) * (
            self.config.dt_ms / 1000.0
        )
        voltage_trace = np.empty((step_count, watch.size), dtype=np.float64)
        drive_trace = np.empty((step_count, watch.size), dtype=np.float64)
        spike_times: list[list[float]] = [
            [] for _ in range(self.connectivity.neuron_count)
        ]

        for step in range(step_count):
            external_indices = input_schedule.get(step)
            spikes = self._advance(external_indices, stimulus.amplitude_mv)
            time_s = step * self.config.dt_ms / 1000.0
            for neuron in spikes:
                spike_times[int(neuron)].append(time_s)
            voltage_trace[step] = self.voltage_mv[watch]
            drive_trace[step] = self.synaptic_drive_mv[watch]

        immutable_times = tuple(tuple(times) for times in spike_times)
        counts = tuple(len(times) for times in immutable_times)
        rates = {
            name: sum(counts[index] for index in indices) / (len(indices) * duration_s)
            for name, indices in self.populations.items()
        }
        readout = NeuralReadout(
            duration_s=duration_s,
            neuron_spike_counts=counts,
            population_rates_hz=rates,
            spike_times_s=immutable_times,
        )
        return SimulationTrace(
            readout=readout,
            watched_indices=watched_indices,
            sample_times_s=sample_times,
            voltage_mv=voltage_trace,
            synaptic_drive_mv=drive_trace,
        )

    def _advance(
        self, external_indices: IntArray | None, external_amplitude_mv: float
    ) -> IntArray:
        cfg = self.config
        active = self.refractory_steps_left == 0
        refractory_indices = np.flatnonzero(~active)
        refractory_voltage = self.voltage_mv[refractory_indices].copy()
        refractory_drive = self.synaptic_drive_mv[refractory_indices].copy()
        self.voltage_mv -= cfg.resting_mv
        self.voltage_mv *= self._membrane_decay
        self.voltage_mv += cfg.resting_mv
        self.voltage_mv += self.synaptic_drive_mv * self._drive_coupling
        self.synaptic_drive_mv *= self._synapse_decay
        self.voltage_mv[refractory_indices] = refractory_voltage
        self.synaptic_drive_mv[refractory_indices] = refractory_drive
        self.refractory_steps_left[~active] -= 1

        spikes = np.flatnonzero(active & (self.voltage_mv > cfg.threshold_mv)).astype(
            np.int64, copy=False
        )
        due_slot = self._step_index % len(self._delay_ring)
        due_sources = self._delay_ring[due_slot]
        self._delay_ring[due_slot] = np.empty(0, dtype=np.int64)
        if due_sources.size:
            self.visited_edges += self.connectivity.accumulate_spikes(
                due_sources,
                self.synaptic_drive_mv,
                amplitudes=cfg.synapse_scale_mv,
            )
            # Brian2's ``unless refractory`` also suppresses synaptic writes to g.
            self.synaptic_drive_mv[refractory_indices] = refractory_drive
        if external_indices is not None:
            receptive = external_indices[active[external_indices]]
            self.voltage_mv[receptive] += external_amplitude_mv

        if spikes.size:
            if cfg.delay_steps == 0:
                self.visited_edges += self.connectivity.accumulate_spikes(
                    spikes,
                    self.synaptic_drive_mv,
                    amplitudes=cfg.synapse_scale_mv,
                )
                self.synaptic_drive_mv[refractory_indices] = refractory_drive
            else:
                delivery_slot = (self._step_index + cfg.delay_steps) % len(
                    self._delay_ring
                )
                self._delay_ring[delivery_slot] = spikes
            self.voltage_mv[spikes] = cfg.reset_mv
            self.synaptic_drive_mv[spikes] = 0.0
            # The spike/reset occurs on the current grid point. Brian2 resumes
            # integration at exactly ``spike_time + refractory``, so only the
            # intervening grid points are skipped.
            self.refractory_steps_left[spikes] = max(0, cfg.refractory_steps - 1)

        self._step_index += 1
        return spikes

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
