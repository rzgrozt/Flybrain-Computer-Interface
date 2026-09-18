"""Deterministic graded reference for the R1-R6 to lamina boundary.

This module is deliberately separate from the whole-network LIF runtime.  It models
only the qualitative early-vision dynamics needed to test whether a tonic,
non-spiking photoreceptor representation fixes the known point-neuron mismatch.
Parameters are explicit engineering priors, not a fitted physiological model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import exp, isclose, isfinite

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GradedEarlyVisionConfig:
    """Time constants and gains for a compact graded early-vision reference."""

    frame_rate_hz: float = 60.0
    neural_dt_ms: float = 0.1
    background_luminance: float = 0.5
    tonic_release_fraction: float = 0.1
    photoreceptor_tau_ms: float = 15.0
    transient_fast_tau_ms: float = 8.0
    transient_slow_tau_ms: float = 40.0
    l3_tau_ms: float = 30.0
    l1_gain_mv: float = 6.0
    l2_gain_mv: float = 6.0
    l2_slow_weight: float = 1.05
    l3_gain_mv: float = 6.0

    def __post_init__(self) -> None:
        numeric = (
            self.frame_rate_hz,
            self.neural_dt_ms,
            self.background_luminance,
            self.tonic_release_fraction,
            self.photoreceptor_tau_ms,
            self.transient_fast_tau_ms,
            self.transient_slow_tau_ms,
            self.l3_tau_ms,
            self.l1_gain_mv,
            self.l2_gain_mv,
            self.l2_slow_weight,
            self.l3_gain_mv,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("graded early-vision parameters must be finite")
        if self.frame_rate_hz <= 0 or self.neural_dt_ms <= 0:
            raise ValueError("frame rate and neural dt must be positive")
        if not 0.0 <= self.background_luminance <= 1.0:
            raise ValueError("background luminance must be in [0, 1]")
        if not 0.0 <= self.tonic_release_fraction < 1.0:
            raise ValueError("tonic release fraction must be in [0, 1)")
        if min(
            self.photoreceptor_tau_ms,
            self.transient_fast_tau_ms,
            self.transient_slow_tau_ms,
            self.l3_tau_ms,
        ) <= 0:
            raise ValueError("graded time constants must be positive")
        if self.transient_fast_tau_ms >= self.transient_slow_tau_ms:
            raise ValueError("transient fast tau must be smaller than slow tau")
        if min(self.l1_gain_mv, self.l2_gain_mv, self.l3_gain_mv) <= 0:
            raise ValueError("lamina gains must be positive")
        if self.l2_slow_weight < 1.0:
            raise ValueError("L2 slow weight must be at least 1")


@dataclass(frozen=True, slots=True)
class GradedEarlyVisionTrace:
    """Neural-grid trace from a spatially uniform graded luminance sequence."""

    sample_times_s: FloatArray
    frame_timestamps_s: tuple[float, ...]
    mean_luminance: tuple[float, ...]
    luminance_by_step: FloatArray
    photoreceptor_state: FloatArray
    histamine_release: FloatArray
    l1_delta_mv: FloatArray
    l2_delta_mv: FloatArray
    l3_delta_mv: FloatArray
    baseline_release: float
    frame_sha256: str

def simulate_graded_early_vision(
    frames: npt.ArrayLike,
    config: GradedEarlyVisionConfig = GradedEarlyVisionConfig(),
) -> GradedEarlyVisionTrace:
    """Simulate tonic graded photoreceptor release and L1/L2/L3 reference signals.

    R1-R6 output is represented as a nonzero tonic histamine-release state whose
    level follows mean luminance through a first-order photoreceptor filter. L1 and
    L2 use a fast-minus-slow temporal filter, while L3 uses a slower sustained
    sign-inverted filter. All lamina outputs are membrane-potential deltas relative
    to an adapted background, not absolute voltages.
    """

    image = np.asarray(frames, dtype=np.float64)
    if image.ndim not in (3, 4) or image.shape[0] == 0:
        raise ValueError("frames must have shape (time, height, width[, channels])")
    if not np.isfinite(image).all() or np.any((image < 0.0) | (image > 1.0)):
        raise ValueError("frame luminance must be finite and normalized to [0, 1]")

    spatial_axes = tuple(range(1, image.ndim))
    luminance = np.asarray(
        np.mean(image, axis=spatial_axes, dtype=np.float64), dtype=np.float64
    )
    frame_period_s = 1.0 / config.frame_rate_hz
    total_duration_s = image.shape[0] * frame_period_s
    dt_s = config.neural_dt_ms / 1000.0
    exact_steps = total_duration_s / dt_s
    step_count = round(exact_steps)
    if not isclose(exact_steps, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("frame sequence duration must align to the neural grid")

    sample_times = np.arange(step_count, dtype=np.float64) * dt_s
    frame_indices = np.minimum(
        np.floor(sample_times * config.frame_rate_hz + 1e-12).astype(np.int64),
        image.shape[0] - 1,
    )
    luminance_steps = luminance[frame_indices].copy()
    frame_times = tuple(frame * frame_period_s for frame in range(image.shape[0]))

    tonic = config.tonic_release_fraction
    background_release = tonic + (1.0 - tonic) * config.background_luminance
    photoreceptor = config.background_luminance
    fast = background_release
    slow = background_release
    l3_state = background_release

    photo_alpha = _alpha(config.neural_dt_ms, config.photoreceptor_tau_ms)
    fast_alpha = _alpha(config.neural_dt_ms, config.transient_fast_tau_ms)
    slow_alpha = _alpha(config.neural_dt_ms, config.transient_slow_tau_ms)
    l3_alpha = _alpha(config.neural_dt_ms, config.l3_tau_ms)

    photoreceptor_trace = np.empty(step_count, dtype=np.float64)
    release_trace = np.empty(step_count, dtype=np.float64)
    l1_trace = np.empty(step_count, dtype=np.float64)
    l2_trace = np.empty(step_count, dtype=np.float64)
    l3_trace = np.empty(step_count, dtype=np.float64)

    for step in range(step_count):
        photoreceptor += photo_alpha * (luminance_steps[step] - photoreceptor)
        release = tonic + (1.0 - tonic) * photoreceptor
        fast += fast_alpha * (release - fast)
        slow += slow_alpha * (release - slow)
        l3_state += l3_alpha * (release - l3_state)

        l1 = -config.l1_gain_mv * (fast - slow)
        l2_filter = (
            fast
            - config.l2_slow_weight * slow
            - (1.0 - config.l2_slow_weight) * background_release
        )
        l2 = -config.l2_gain_mv * l2_filter
        l3 = -config.l3_gain_mv * (l3_state - background_release)

        photoreceptor_trace[step] = photoreceptor
        release_trace[step] = release
        l1_trace[step] = l1
        l2_trace[step] = l2
        l3_trace[step] = l3

    canonical = np.asarray(image, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))

    return GradedEarlyVisionTrace(
        sample_times_s=sample_times,
        frame_timestamps_s=frame_times,
        mean_luminance=tuple(float(value) for value in luminance),
        luminance_by_step=luminance_steps,
        photoreceptor_state=photoreceptor_trace,
        histamine_release=release_trace,
        l1_delta_mv=l1_trace,
        l2_delta_mv=l2_trace,
        l3_delta_mv=l3_trace,
        baseline_release=background_release,
        frame_sha256=digest.hexdigest(),
    )



@dataclass(frozen=True, slots=True)
class GradedPhotoreceptorTrace:
    """Parallel graded photoreceptor channels on the neural time grid."""

    sample_times_s: FloatArray
    frame_timestamps_s: tuple[float, ...]
    luminance_by_step: FloatArray
    photoreceptor_state: FloatArray
    histamine_release: FloatArray
    histamine_release_delta: FloatArray
    baseline_release: float
    frame_sha256: str

    def __post_init__(self) -> None:
        arrays = (
            self.luminance_by_step,
            self.photoreceptor_state,
            self.histamine_release,
            self.histamine_release_delta,
        )
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("graded photoreceptor arrays must be two-dimensional")
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("graded photoreceptor arrays must have matching shapes")
        if arrays[0].shape[0] != self.sample_times_s.size:
            raise ValueError("graded photoreceptor time axis does not match samples")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("graded photoreceptor arrays must be finite")


def simulate_graded_photoreceptor_channels(
    intensity_by_frame: npt.ArrayLike,
    config: GradedEarlyVisionConfig = GradedEarlyVisionConfig(),
) -> GradedPhotoreceptorTrace:
    """Simulate independent tonic graded R1-R6 release channels.

    Input has shape (frames, channels). Every channel starts adapted to the same
    configured background luminance, then follows sampled luminance through the
    photoreceptor time constant. Release delta is relative to the tonic adapted
    background and is exactly zero for an all-background sequence.
    """

    intensity = np.asarray(intensity_by_frame, dtype=np.float64)
    if intensity.ndim != 2 or intensity.shape[0] == 0 or intensity.shape[1] == 0:
        raise ValueError(
            "intensity_by_frame must have shape (frames, channels) with nonzero axes"
        )
    if not np.isfinite(intensity).all() or np.any(
        (intensity < 0.0) | (intensity > 1.0)
    ):
        raise ValueError(
            "photoreceptor intensity must be finite and normalized to [0, 1]"
        )

    frame_period_s = 1.0 / config.frame_rate_hz
    total_duration_s = intensity.shape[0] * frame_period_s
    dt_s = config.neural_dt_ms / 1000.0
    exact_steps = total_duration_s / dt_s
    step_count = round(exact_steps)
    if not isclose(exact_steps, step_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("frame sequence duration must align to the neural grid")

    sample_times = np.arange(step_count, dtype=np.float64) * dt_s
    frame_indices = np.minimum(
        np.floor(sample_times * config.frame_rate_hz + 1e-12).astype(np.int64),
        intensity.shape[0] - 1,
    )
    luminance_steps = intensity[frame_indices].copy()
    frame_times = tuple(frame * frame_period_s for frame in range(intensity.shape[0]))

    tonic = config.tonic_release_fraction
    baseline_release = tonic + (1.0 - tonic) * config.background_luminance
    state = np.full(
        intensity.shape[1],
        config.background_luminance,
        dtype=np.float64,
    )
    alpha = _alpha(config.neural_dt_ms, config.photoreceptor_tau_ms)
    state_trace = np.empty_like(luminance_steps)
    release_trace = np.empty_like(luminance_steps)
    release_delta = np.empty_like(luminance_steps)

    for step in range(step_count):
        state += alpha * (luminance_steps[step] - state)
        release = tonic + (1.0 - tonic) * state
        state_trace[step] = state
        release_trace[step] = release
        release_delta[step] = release - baseline_release

    canonical = np.asarray(intensity, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))

    for array in (luminance_steps, state_trace, release_trace, release_delta):
        array.flags.writeable = False
    sample_times.flags.writeable = False
    return GradedPhotoreceptorTrace(
        sample_times_s=sample_times,
        frame_timestamps_s=frame_times,
        luminance_by_step=luminance_steps,
        photoreceptor_state=state_trace,
        histamine_release=release_trace,
        histamine_release_delta=release_delta,
        baseline_release=baseline_release,
        frame_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class GradedEarlyVisionChannelTrace:
    """Parallel graded R1-R6 and lamina reference channels."""

    photoreceptor: GradedPhotoreceptorTrace
    l1_delta_mv: FloatArray
    l2_delta_mv: FloatArray
    l3_delta_mv: FloatArray

    def __post_init__(self) -> None:
        expected = self.photoreceptor.histamine_release.shape
        for array in (self.l1_delta_mv, self.l2_delta_mv, self.l3_delta_mv):
            if array.shape != expected:
                raise ValueError("graded lamina arrays must match photoreceptor shape")
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError("graded lamina arrays must be finite 2-D arrays")


def simulate_graded_early_vision_channels(
    intensity_by_frame: npt.ArrayLike,
    config: GradedEarlyVisionConfig = GradedEarlyVisionConfig(),
) -> GradedEarlyVisionChannelTrace:
    """Simulate parallel graded photoreceptor and L1/L2/L3 reference channels."""

    photo = simulate_graded_photoreceptor_channels(intensity_by_frame, config)
    channel_count = photo.histamine_release.shape[1]
    fast = np.full(channel_count, photo.baseline_release, dtype=np.float64)
    slow = np.full(channel_count, photo.baseline_release, dtype=np.float64)
    l3_state = np.full(channel_count, photo.baseline_release, dtype=np.float64)
    fast_alpha = _alpha(config.neural_dt_ms, config.transient_fast_tau_ms)
    slow_alpha = _alpha(config.neural_dt_ms, config.transient_slow_tau_ms)
    l3_alpha = _alpha(config.neural_dt_ms, config.l3_tau_ms)

    l1_trace = np.empty_like(photo.histamine_release)
    l2_trace = np.empty_like(photo.histamine_release)
    l3_trace = np.empty_like(photo.histamine_release)
    for step in range(photo.histamine_release.shape[0]):
        release = photo.histamine_release[step]
        fast += fast_alpha * (release - fast)
        slow += slow_alpha * (release - slow)
        l3_state += l3_alpha * (release - l3_state)

        l1_trace[step] = -config.l1_gain_mv * (fast - slow)
        l2_filter = (
            fast
            - config.l2_slow_weight * slow
            - (1.0 - config.l2_slow_weight) * photo.baseline_release
        )
        l2_trace[step] = -config.l2_gain_mv * l2_filter
        l3_trace[step] = -config.l3_gain_mv * (
            l3_state - photo.baseline_release
        )

    for array in (l1_trace, l2_trace, l3_trace):
        array.flags.writeable = False
    return GradedEarlyVisionChannelTrace(
        photoreceptor=photo,
        l1_delta_mv=l1_trace,
        l2_delta_mv=l2_trace,
        l3_delta_mv=l3_trace,
    )

def _alpha(dt_ms: float, tau_ms: float) -> float:
    return 1.0 - exp(-dt_ms / tau_ms)