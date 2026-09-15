"""Deterministic uniform-luminance engineering transduction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import floor, isfinite

import numpy as np
import numpy.typing as npt

from flybrain_interface.sensory.spikes import DeterministicSpikeInput


@dataclass(frozen=True, slots=True)
class UniformLuminanceConfig:
    """Explicit frame-to-pulse approximation, not photoreceptor physiology."""

    frame_rate_hz: float
    max_pulses_per_frame: int
    pulse_amplitude_mv: float
    neural_dt_ms: float
    start_s: float = 0.0

    def __post_init__(self) -> None:
        numeric = (
            self.frame_rate_hz,
            self.pulse_amplitude_mv,
            self.neural_dt_ms,
            self.start_s,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("uniform-luminance parameters must be finite")
        if self.frame_rate_hz <= 0 or self.neural_dt_ms <= 0:
            raise ValueError("frame rate and neural dt must be positive")
        if self.max_pulses_per_frame <= 0:
            raise ValueError("max_pulses_per_frame must be positive")
        if self.pulse_amplitude_mv <= 0 or self.start_s < 0:
            raise ValueError("pulse amplitude must be positive and start non-negative")
        available_steps = floor(1000.0 / (self.frame_rate_hz * self.neural_dt_ms))
        if self.max_pulses_per_frame > available_steps:
            raise ValueError("pulse count exceeds distinct neural steps in one frame")


@dataclass(frozen=True, slots=True)
class UniformLuminanceEncoding:
    stimulus: DeterministicSpikeInput
    frame_timestamps_s: tuple[float, ...]
    mean_luminance: tuple[float, ...]
    pulse_counts_per_frame: tuple[int, ...]
    frame_sha256: str
    quantization: str


def encode_uniform_luminance(
    frames: npt.ArrayLike,
    neuron_indices: tuple[int, ...],
    config: UniformLuminanceConfig,
) -> UniformLuminanceEncoding:
    """Collapse each image to mean luminance and emit deterministic pulse density.

    The frame clock is sampled independently and event times are then rounded to the
    nearest neural step (half upward). Spatial structure is intentionally discarded
    because no validated R1-R6 receptive-field mapping is supplied.
    """

    image = np.asarray(frames, dtype=np.float64)
    if image.ndim not in (3, 4) or image.shape[0] == 0:
        raise ValueError("frames must have shape (time, height, width[, channels])")
    if not np.isfinite(image).all() or np.any((image < 0.0) | (image > 1.0)):
        raise ValueError("frame luminance must be finite and normalized to [0, 1]")
    if not neuron_indices or len(neuron_indices) != len(set(neuron_indices)):
        raise ValueError("neuron_indices must be non-empty and unique")
    if any(index < 0 for index in neuron_indices):
        raise ValueError("neuron indices must be non-negative")

    spatial_axes = tuple(range(1, image.ndim))
    luminance = np.asarray(
        np.mean(image, axis=spatial_axes, dtype=np.float64), dtype=np.float64
    )
    frame_period_s = 1.0 / config.frame_rate_hz
    dt_s = config.neural_dt_ms / 1000.0
    frame_times = tuple(
        config.start_s + frame * frame_period_s for frame in range(image.shape[0])
    )
    pulse_counts = tuple(
        floor(float(value) * config.max_pulses_per_frame + 0.5) for value in luminance
    )
    event_steps: list[int] = []
    event_neurons: list[int] = []
    for frame, pulse_count in enumerate(pulse_counts):
        frame_steps: list[int] = []
        for pulse in range(pulse_count):
            raw_time = frame_times[frame] + (
                (pulse + 0.5) * frame_period_s / pulse_count
            )
            step = floor(raw_time / dt_s + 0.5)
            frame_steps.append(step)
            event_steps.extend([step] * len(neuron_indices))
            event_neurons.extend(neuron_indices)
        if len(frame_steps) != len(set(frame_steps)):
            raise ValueError("frame pulses collided after neural-grid quantization")

    canonical = np.asarray(image, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return UniformLuminanceEncoding(
        stimulus=DeterministicSpikeInput(
            neuron_indices=tuple(event_neurons),
            times_s=tuple(step * dt_s for step in event_steps),
            amplitude_mv=config.pulse_amplitude_mv,
        ),
        frame_timestamps_s=frame_times,
        mean_luminance=tuple(float(value) for value in luminance),
        pulse_counts_per_frame=pulse_counts,
        frame_sha256=digest.hexdigest(),
        quantization="nearest neural step, half upward",
    )
