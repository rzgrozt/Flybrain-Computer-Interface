from __future__ import annotations

import pytest

from flybrain_interface.simulation.config import ShiuLIFConfig


def test_shiu_config_converts_grid_aligned_durations_to_steps() -> None:
    config = ShiuLIFConfig()

    assert config.delay_steps == 18
    assert config.refractory_steps == 22


def test_shiu_config_rejects_nonfinite_and_off_grid_parameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        ShiuLIFConfig(threshold_mv=float("nan"))
    with pytest.raises(ValueError, match="integer multiple"):
        _ = ShiuLIFConfig(delay_ms=1.85).delay_steps
    with pytest.raises(ValueError, match="below threshold"):
        ShiuLIFConfig(tonic_bias_mv=7.0)
