"""Real MaleCNS two-/three-hop observations remain sampled, never inferred.

The public verified artifact and local normalized MaleCNS data are required.
No neural transmission or motor causality is inferred from anatomical routes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybrain_interface.panel.models import (
    ExperimentConfig,
    PathwayObservationConfig,
    StimulusConfig,
)
from flybrain_interface.panel.pathway_observation import resolve_observation
from flybrain_interface.panel.pathways_v2 import PATHWAY_ARTIFACT, pathway_catalog
from flybrain_interface.panel.worker import WorkerSession

DATA = Path(__file__).resolve().parents[2] / "data/processed/malecns-v1.0"


@pytest.mark.skipif(
    not PATHWAY_ARTIFACT.is_file() or not (DATA / "csr_indices.npy").is_file(),
    reason="local normalized MaleCNS and verified anatomy artifact required",
)
def test_real_worker_can_switch_bounded_two_and_three_hop_observation(
    tmp_path: Path,
) -> None:
    catalog = pathway_catalog()
    two_target = next(
        item for item in catalog["targets"] if item["target_instance"] == "DNp09_L"
    )
    three_target = next(
        item for item in catalog["targets"] if item["target_instance"] == "DNa02_L"
    )
    two = PathwayObservationConfig(
        target_neuron_index=two_target["target_neuron_index"], path_index=0
    )
    three = PathwayObservationConfig(
        target_neuron_index=three_target["target_neuron_index"], path_index=0
    )
    two_indices = resolve_observation(two)["neuron_indices"]
    three_indices = resolve_observation(three)["neuron_indices"]
    assert len(two_indices) == 3
    assert len(three_indices) == 4

    config = ExperimentConfig(
        duration_s=0.04,
        chunk_duration_s=0.02,
        backend="numba",
        subnormal_drive_policy="zero",
        watch_indices=[0],
        visualization_indices=[0],
        populations=[],
        pathway_observation=two,
        stimulus=StimulusConfig(
            neuron_indices=[two_indices[0]],
            start_s=0.0,
            stop_s=0.04,
            interval_ms=10.0,
            amplitude_mv=8.0,
        ),
    )
    session = WorkerSession.create(config, DATA, tmp_path)
    session.advance()
    first = session.snapshot("sample")
    sample = first["pathway_measurement"]
    assert sample is not None
    assert sample["source"] == "simulated_neural_measurement"
    assert sample["sampled_chunk"] == 1
    assert sample["neuron_indices"] == two_indices
    assert sample["bin_duration_s"] == pytest.approx(0.02)
    assert len(sample["spike_counts"]) == 3
    assert all(
        isinstance(count, int) and count >= 0 for count in sample["spike_counts"]
    )
    np.testing.assert_allclose(
        sample["voltage_mv"], session.simulator.voltage_mv[two_indices]
    )
    np.testing.assert_allclose(
        sample["synaptic_drive_mv"],
        session.simulator.synaptic_drive_mv[two_indices],
    )
    assert len(first["voltage_mv"]) == len(first["watch_indices"]) == 1
    assert first.get("pathway_activation") is None

    # The backend ignores any caller-supplied neuron list and resolves the
    # switched route from immutable verified anatomy.
    session.configure_pathway({**three.model_dump(), "neuron_indices": [999_999]})
    assert session.snapshot("switched")["pathway_measurement"] is None
    session.advance()
    second = session.snapshot("sample")
    next_sample = second["pathway_measurement"]
    assert next_sample is not None
    assert next_sample["sampled_chunk"] == 2
    assert next_sample["neuron_indices"] == three_indices
    assert len(next_sample["spike_counts"]) == 4
    assert all(
        isinstance(count, int) and count >= 0 for count in next_sample["spike_counts"]
    )
    np.testing.assert_allclose(
        next_sample["voltage_mv"], session.simulator.voltage_mv[three_indices]
    )
    np.testing.assert_allclose(
        next_sample["synaptic_drive_mv"],
        session.simulator.synaptic_drive_mv[three_indices],
    )
    assert second["watch_indices"] == [0]
    session.configure_pathway(None)
    assert session.snapshot("cleared")["pathway_measurement"] is None
