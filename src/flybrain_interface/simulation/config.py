"""Shared, unit-explicit neural simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True, slots=True)
class ShiuLIFConfig:
    """Published Shiu et al. baseline parameters in milliseconds/millivolts."""

    resting_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    synapse_scale_mv: float = 0.275
    dt_ms: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.resting_mv,
            self.reset_mv,
            self.threshold_mv,
            self.membrane_tau_ms,
            self.synapse_tau_ms,
            self.refractory_ms,
            self.delay_ms,
            self.synapse_scale_mv,
            self.dt_ms,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("LIF parameters must be finite")
        if self.threshold_mv <= self.resting_mv:
            raise ValueError("threshold_mv must exceed resting_mv")
        if self.membrane_tau_ms <= 0 or self.synapse_tau_ms <= 0:
            raise ValueError("time constants must be positive")
        if self.refractory_ms < 0 or self.delay_ms < 0 or self.dt_ms <= 0:
            raise ValueError("refractory/delay must be non-negative and dt positive")
        if self.synapse_scale_mv <= 0:
            raise ValueError("synapse_scale_mv must be positive")
        if isclose(self.membrane_tau_ms, self.synapse_tau_ms):
            raise ValueError(
                "equal membrane and synapse time constants are unsupported"
            )

    @property
    def delay_steps(self) -> int:
        return _whole_steps(self.delay_ms, self.dt_ms, "delay_ms")

    @property
    def refractory_steps(self) -> int:
        return _whole_steps(self.refractory_ms, self.dt_ms, "refractory_ms")


def _whole_steps(duration_ms: float, dt_ms: float, label: str) -> int:
    steps = round(duration_ms / dt_ms)
    if not isclose(steps * dt_ms, duration_ms, abs_tol=1e-12):
        raise ValueError(f"{label} must be an integer multiple of dt_ms")
    return int(steps)
