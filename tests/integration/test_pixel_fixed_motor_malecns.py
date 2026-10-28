"""Canonical pixel baseline on local, checksum-verified MaleCNS data."""

from pathlib import Path

import pytest

from flybrain_interface.experiments.pixel_fixed_motor import run_validation

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/malecns-v1.0"


@pytest.mark.skipif(
    not (DATA / "csr_indices.npy").is_file()
    or not (ROOT / "data/processed/eyemap-view-directions-v1.csv").is_file()
    or not (ROOT / "data/raw/optic-column-type-assignments-v1.0.xlsx").is_file(),
    reason="local MaleCNS and measured eye/column data required",
)
def test_full_graph_pixels_preserve_integrity_and_controls() -> None:
    result = run_validation(DATA)
    assert all(result["integrity_checks"].values())
    assert result["signal_checks"]["pixels_reach_downstream"]
    assert len(result["conditions"]["pixels"]["ticks"]) == 10
    assert result["dataset"]["neurons"] == 166700
    assert len(result["dataset"]["verified_files"]) == 8
    # Silence at motor outputs is a measured outcome, not an integrity failure.
    assert not result["runtime"]["graded_relays_enabled"]
    assert not result["runtime"]["plasticity_enabled"]
    assert not result["runtime"]["os_actions_enabled"]
