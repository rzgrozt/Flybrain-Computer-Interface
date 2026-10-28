"""Real anatomical paths, full-graph interventions and induced reference gates."""

from pathlib import Path

import pytest

from flybrain_interface.experiments.pixel_propagation_audit import run_audit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/malecns-v1.0"


@pytest.mark.skipif(
    not (DATA / "csr_indices.npy").is_file()
    or not (ROOT / "data/processed/eyemap-view-directions-v1.csv").is_file()
    or not (ROOT / "data/raw/optic-column-type-assignments-v1.0.xlsx").is_file(),
    reason="local MaleCNS and measured eye/column data required",
)
def test_pixel_pathways_and_test_pulses_match_reference() -> None:
    result = run_audit(DATA)
    assert all(result["integrity_checks"].values())
    assert 0 < result["pathway_watch_count"] <= 4096
    assert len(result["pathway_responses"]) == 10
    assert len(result["selected_routes"]) == 2
    assert len(result["interventions"]) == 5
    assert len(result["reference_comparisons"]) == 10
    for comparison in result["reference_comparisons"]:
        assert comparison["equivalent"]
        assert comparison["induced_edges"] > 0
    for condition in result["conditions"].values():
        assert len(condition["watched_neurons"]) >= result["pathway_watch_count"]
    assert not result["runtime"]["graded_relays_enabled"]
