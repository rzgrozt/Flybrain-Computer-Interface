"""Comparable outputs from reference and runtime neural simulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flybrain_interface.contracts import NeuralReadout

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ChunkRecording:
    """Bounded recording requested for one state-preserving runtime chunk."""

    watched_indices: tuple[int, ...] = ()
    include_neuron_counts: bool = False
    max_spike_events: int = 0

    def __post_init__(self) -> None:
        if self.max_spike_events < 0:
            raise ValueError("max_spike_events cannot be negative")


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Finite observations from one chunk; no history is retained by the runtime."""

    start_step: int
    end_step: int
    dt_ms: float
    total_spikes: int
    population_rates_hz: dict[str, float]
    neuron_spike_counts: IntArray | None
    spike_neuron_indices: IntArray
    spike_times_s: FloatArray
    dropped_spike_events: int
    watched_indices: tuple[int, ...]
    sample_times_s: FloatArray
    voltage_mv: FloatArray
    synaptic_drive_mv: FloatArray

    def __post_init__(self) -> None:
        step_count = self.end_step - self.start_step
        expected_trace_shape = (step_count, len(self.watched_indices))
        if self.start_step < 0 or step_count <= 0:
            raise ValueError("chunk step range must be positive")
        if self.total_spikes < 0 or self.dropped_spike_events < 0:
            raise ValueError("spike totals cannot be negative")
        if self.spike_neuron_indices.shape != self.spike_times_s.shape:
            raise ValueError("recorded spike arrays must have equal shape")
        if self.sample_times_s.shape != (step_count,):
            raise ValueError("sample_times_s must contain one value per step")
        if (
            self.voltage_mv.shape != expected_trace_shape
            or self.synaptic_drive_mv.shape != expected_trace_shape
        ):
            raise ValueError("chunk traces must be time-by-watched-neuron")

    @property
    def duration_s(self) -> float:
        return (self.end_step - self.start_step) * self.dt_ms / 1000.0


@dataclass(frozen=True, slots=True)
class SimulationTrace:
    readout: NeuralReadout
    watched_indices: tuple[int, ...]
    sample_times_s: FloatArray
    voltage_mv: FloatArray
    synaptic_drive_mv: FloatArray

    def __post_init__(self) -> None:
        expected = (self.sample_times_s.size, len(self.watched_indices))
        if self.sample_times_s.ndim != 1:
            raise ValueError("sample_times_s must be one-dimensional")
        if (
            self.voltage_mv.shape != expected
            or self.synaptic_drive_mv.shape != expected
        ):
            raise ValueError("trace matrices must be time-by-watched-neuron")
