"""Deterministic sparse projected-drive inputs for non-spiking source populations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(slots=True)
class ProjectedDriveChannel:
    """One preprojected source population with a per-step signed amplitude.

    signed_contact_weights already contains the transmitter sign and summed
    MaleCNS contact counts for each unique target. amplitudes_mv is the
    per-neural-step amplitude multiplier applied to those weights.
    """

    label: str
    target_indices: IntArray
    signed_contact_weights: FloatArray
    amplitudes_mv: FloatArray
    anatomical_source_count: int
    anatomical_edge_count: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("projected drive label cannot be empty")
        if self.anatomical_source_count <= 0 or self.anatomical_edge_count < 0:
            raise ValueError("projected drive anatomical counts are invalid")

        targets = np.asarray(self.target_indices, dtype=np.int64).copy()
        weights = np.asarray(self.signed_contact_weights, dtype=np.float64).copy()
        amplitudes = np.asarray(self.amplitudes_mv, dtype=np.float64).copy()
        if targets.ndim != 1 or weights.ndim != 1 or amplitudes.ndim != 1:
            raise ValueError("projected drive arrays must be one-dimensional")
        if targets.shape != weights.shape:
            raise ValueError("projected drive target and weight arrays must align")
        if targets.size and np.unique(targets).size != targets.size:
            raise ValueError("projected drive targets must be unique per channel")
        if not np.isfinite(weights).all() or not np.isfinite(amplitudes).all():
            raise ValueError("projected drive values must be finite")
        if np.any(weights == 0.0):
            raise ValueError("projected drive weights must exclude zero entries")
        if amplitudes.size == 0:
            raise ValueError("projected drive amplitude trace cannot be empty")

        targets.flags.writeable = False
        weights.flags.writeable = False
        amplitudes.flags.writeable = False
        self.target_indices = targets
        self.signed_contact_weights = weights
        self.amplitudes_mv = amplitudes

    @property
    def step_count(self) -> int:
        return int(self.amplitudes_mv.size)


@dataclass(slots=True)
class DeterministicProjectedDriveInput:
    """Chunk-relative projected drive traces on the neural time grid."""

    channels: tuple[ProjectedDriveChannel, ...]
    dt_ms: float

    def __post_init__(self) -> None:
        if not isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise ValueError("projected drive dt_ms must be finite and positive")
        if not self.channels:
            raise ValueError("projected drive must contain at least one channel")
        expected = self.channels[0].step_count
        if any(channel.step_count != expected for channel in self.channels[1:]):
            raise ValueError("projected drive channels must have equal trace lengths")

    @property
    def step_count(self) -> int:
        return self.channels[0].step_count
