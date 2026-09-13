"""Deterministic language normalization for simple development goals."""

from __future__ import annotations

from dataclasses import dataclass

from flybrain_interface.contracts import StructuredGoal


@dataclass(frozen=True, slots=True)
class LiteralGoalTranslator:
    """Treat text as a concept without interpreting it into an action sequence."""

    desired_state: str = "present"

    def translate(self, instruction: str) -> StructuredGoal:
        normalized = " ".join(instruction.casefold().split())
        return StructuredGoal(concept=normalized, desired_state=self.desired_state)
