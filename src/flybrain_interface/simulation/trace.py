"""Comparable outputs from reference and runtime neural simulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flybrain_interface.contracts import NeuralReadout

FloatArray = npt.NDArray[np.float64]


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
