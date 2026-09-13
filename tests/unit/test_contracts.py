"""Tests for safety-critical subsystem contracts."""

import pytest

from flybrain_interface.contracts import NeuralReadout, StructuredGoal


def test_goal_has_no_motor_or_coordinate_fields() -> None:
    goal = StructuredGoal(concept="spreadsheet application", desired_state="open")

    assert set(goal.__slots__) == {"concept", "desired_state", "qualifiers"}


def test_neural_readout_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        NeuralReadout(
            duration_s=0.1,
            neuron_spike_counts=(-1,),
            population_rates_hz={},
        )
