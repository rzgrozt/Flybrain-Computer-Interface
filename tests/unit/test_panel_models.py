from __future__ import annotations

import pytest
from pydantic import ValidationError

from flybrain_interface.panel.models import ExperimentConfig, StimulusConfig


def test_experiment_defaults_to_exact_subnormal_preservation() -> None:
    config = ExperimentConfig()

    assert config.subnormal_drive_policy == "preserve"
    assert config.watch_indices == [0]
    assert config.visualization_indices == [0]


def test_experiment_rejects_unbounded_or_ambiguous_targets() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(duration_s=61.0)
    with pytest.raises(ValidationError):
        ExperimentConfig(watch_indices=list(range(33)))
    with pytest.raises(ValidationError):
        ExperimentConfig(stimulus=StimulusConfig(neuron_indices=[2, 2]))
    with pytest.raises(ValidationError):
        ExperimentConfig(visualization_indices=list(range(4097)))


def test_stimulus_window_must_fit_experiment() -> None:
    with pytest.raises(ValidationError, match="stimulus window"):
        ExperimentConfig(
            duration_s=0.1,
            stimulus=StimulusConfig(stop_s=0.2),
        )
