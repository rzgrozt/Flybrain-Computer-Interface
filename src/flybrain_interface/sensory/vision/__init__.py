"""Deterministic visual sensory encoding."""

from flybrain_interface.sensory.vision.graded import (
    GradedEarlyVisionConfig,
    GradedEarlyVisionTrace,
    simulate_graded_early_vision,
)
from flybrain_interface.sensory.vision.uniform import (
    UniformLuminanceConfig,
    UniformLuminanceEncoding,
    encode_uniform_luminance,
)

__all__ = [
    "GradedEarlyVisionConfig",
    "GradedEarlyVisionTrace",
    "UniformLuminanceConfig",
    "UniformLuminanceEncoding",
    "encode_uniform_luminance",
    "simulate_graded_early_vision",
]
