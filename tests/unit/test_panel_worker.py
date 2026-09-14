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
