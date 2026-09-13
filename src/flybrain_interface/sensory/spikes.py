"""Explicit synthetic neural stimulation for reference validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeterministicSpikeInput:
    """Artificial input spikes; times are seconds and amplitudes are millivolts."""

    neuron_indices: tuple[int, ...]
    times_s: tuple[float, ...]
    amplitude_mv: float = 8.0

    def __post_init__(self) -> None:
        if len(self.neuron_indices) != len(self.times_s):
            raise ValueError("neuron_indices and times_s must have equal length")
        if any(time < 0 for time in self.times_s):
            raise ValueError("input spike times cannot be negative")
