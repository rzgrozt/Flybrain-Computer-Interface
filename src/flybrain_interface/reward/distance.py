"""Deterministic teacher reward for early cursor-target experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from flybrain_interface.contracts import RewardSignal


@dataclass(frozen=True, slots=True)
class DistanceState:
    previous_cursor: tuple[float, float]
    cursor: tuple[float, float]
    target: tuple[float, float]
    success_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class DistanceImprovementEvaluator:
    """Expose hidden geometry only as scalar progress and completion."""

    def evaluate(self, state: DistanceState) -> RewardSignal:
        previous = hypot(
            state.previous_cursor[0] - state.target[0],
            state.previous_cursor[1] - state.target[1],
        )
        current = hypot(
            state.cursor[0] - state.target[0],
            state.cursor[1] - state.target[1],
        )
        return RewardSignal(
            value=previous - current,
            terminal=current <= state.success_radius,
            reason=(
                "target reached"
                if current <= state.success_radius
                else "distance change"
            ),
        )
