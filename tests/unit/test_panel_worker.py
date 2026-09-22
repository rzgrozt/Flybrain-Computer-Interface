from __future__ import annotations

import json
from pathlib import Path
from queue import Queue

import numpy as np

from flybrain_interface.panel.models import ExperimentConfig, StimulusConfig
from flybrain_interface.panel.worker import WorkerSession, publish_latest
from flybrain_interface.simulation.runtime import SparseLIFSimulator


class EmptyConnectivity:
    neuron_count = 4
    edge_count = 0
    directory = Path("/read-only/malecns-v1.0")
    sign_policy = type("SignPolicy", (), {"name": "test-sign-policy"})()

    def accumulate_spikes(
        self,
        spiking_indices: object,
        destination: np.ndarray[tuple[int], np.dtype[np.float64]],
        *,
        amplitudes: object | None = None,
    ) -> int:
        return 0


def make_session(tmp_path: Path) -> WorkerSession:
    config = ExperimentConfig(
        duration_s=0.04,
        chunk_duration_s=0.02,
        watch_indices=[0],
        populations=[],
        stimulus=StimulusConfig(
            neuron_indices=[0],
            start_s=0,
            stop_s=0.04,
            interval_ms=10,
            amplitude_mv=8,
        ),
    )
    simulator = SparseLIFSimulator(EmptyConnectivity(), backend="numpy")
    output = tmp_path / "experiment"
    output.mkdir()
    return WorkerSession(
        config,
        simulator,
        simulator.connectivity,  # type: ignore[arg-type]
        output,
        {"dataset": "male-cns:v1.0"},
    )


def test_pause_resume_boundary_preserves_neural_and_delay_state(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.advance()
    voltage = session.simulator.voltage_mv.copy()
    drive = session.simulator.synaptic_drive_mv.copy()
    pending = session.simulator.pending_delayed_events
    step = session.simulator.current_step

    session.pause()
    assert session.simulator.current_step == step
    assert np.array_equal(session.simulator.voltage_mv, voltage)
    assert np.array_equal(session.simulator.synaptic_drive_mv, drive)
    assert session.simulator.pending_delayed_events == pending

    session.resume()
    session.advance()
    assert session.status == "completed"
    assert session.simulator.current_time_s == 0.04


def test_deterministic_stimulus_and_manifest_record_policy(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    first = session._chunk_stimulus(0.02)
    assert first.neuron_indices == (0, 0)
    assert first.times_s == (0.0, 0.01)
    session.advance()
    second = session._chunk_stimulus(0.02)
    assert second.times_s == (0.0, 0.01)

    session.finish("completed")
    manifest = json.loads((session.output_path / "manifest.json").read_text())
    assert manifest["configuration"]["subnormal_drive_policy"] == "preserve"
    assert manifest["configuration"]["stimulus"]["neuron_indices"] == [0]
    assert manifest["recording"]["watchlist_limit"] == 32
    assert manifest["dataset"]["name"] == "male-cns:v1.0"


def test_latest_telemetry_replaces_slow_consumer_frame() -> None:
    sink: Queue[dict[str, object]] = Queue(maxsize=1)
    publish_latest(sink, {"sequence": 1})  # type: ignore[arg-type]
    publish_latest(sink, {"sequence": 2})  # type: ignore[arg-type]

    event = sink.get_nowait()
    assert event["sequence"] == 2
    assert event["ipc_telemetry_dropped_total"] == 1


def test_dynamic_pathway_observation_records_only_watched_neurons(
    tmp_path: Path, monkeypatch
) -> None:
    import flybrain_interface.panel.worker as worker

    session = make_session(tmp_path)
    routes = {
        2: [0, 1, 2],
        3: [1, 2, 3],
    }

    def fake_resolve(config):
        indices = routes[config.target_neuron_index]
        return {
            "target_neuron_index": config.target_neuron_index,
            "path_index": 0,
            "neuron_indices": indices,
            "body_ids": [str(index) for index in indices],
            "hop_count": 2,
            "source": "verified_connectome_anatomy",
        }

    monkeypatch.setattr(worker, "resolve_observation", fake_resolve)
    session.configure_pathway({"target_neuron_index": 2, "path_index": 0})
    assert session.snapshot("selected")["pathway_measurement"] is None
    session.advance()
    frame = session.snapshot("sample")
    sample = frame["pathway_measurement"]
    assert frame["pathway_selection"]["neuron_indices"] == [0, 1, 2]
    assert sample["source"] == "simulated_neural_measurement"
    assert sample["sampled_chunk"] == 1
    assert sample["simulated_time_s"] == 0.02
    assert sample["bin_duration_s"] == 0.02
    assert len(sample["voltage_mv"]) == len(sample["synaptic_drive_mv"]) == 3
    assert len(sample["spike_counts"]) == 3
    assert all(value >= 0 for value in sample["spike_counts"])
    assert sample["population_spikes"] == sum(sample["spike_counts"])
    assert len(frame["watch_indices"]) == len(frame["voltage_mv"]) == 1

    # Old traces must not masquerade as measurements for a newly selected route.
    session.configure_pathway(
        {
            "target_neuron_index": 3,
            "path_index": 0,
            "neuron_indices": [999],
            "body_ids": ["wrong"],
            "source": "verified_connectome_anatomy",
        }
    )
    assert session.snapshot("changed")["pathway_measurement"] is None
    session.advance()
    replaced = session.snapshot("sample")["pathway_measurement"]
    assert replaced["neuron_indices"] == [1, 2, 3]
    assert replaced["sampled_chunk"] == 2
    session.configure_pathway(None)
    assert session.snapshot("cleared")["pathway_measurement"] is None
    assert session.snapshot("cleared")["pathway_selection"] is None
