"""Compare neural-loop timing across bounded panel-consumer conditions."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from flybrain_interface.panel.app import TelemetryHub
from flybrain_interface.panel.controller import PanelController
from flybrain_interface.panel.models import ExperimentConfig, StimulusConfig
from flybrain_interface.panel.worker import WorkerSession

Mode = Literal["headless", "disconnected", "connected", "slow"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.repetitions <= 0:
        parser.error("--repetitions must be positive")
    config = ExperimentConfig(
        duration_s=arguments.duration_s,
        chunk_duration_s=min(0.02, arguments.duration_s),
        telemetry_hz=30,
        watch_indices=[0, 1],
        populations=[],
        stimulus=StimulusConfig(
            neuron_indices=[0, 1],
            start_s=0,
            stop_s=min(0.05, arguments.duration_s),
            interval_ms=5,
            amplitude_mv=8,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="flybrain-panel-benchmark-") as temp:
        output = Path(temp)
        results = {
            mode: [
                run_condition(mode, config, arguments.data_directory, output)
                for _ in range(arguments.repetitions)
            ]
            for mode in ("headless", "disconnected", "connected", "slow")
        }
    baseline = statistics.median(
        item["worker_loop_seconds"] for item in results["headless"]
    )
    summary = {
        mode: {
            "median_worker_loop_seconds": statistics.median(
                item["worker_loop_seconds"] for item in runs
            ),
            "median_wall_seconds": statistics.median(
                item["wall_seconds"] for item in runs
            ),
            "worker_loop_change_percent_vs_headless": 100
            * (
                statistics.median(item["worker_loop_seconds"] for item in runs)
                / baseline
                - 1
            ),
            "browser_frames_dropped": sum(
                int(item["browser_frames_dropped"]) for item in runs
            ),
            "ipc_telemetry_dropped_total": sum(
                int(item["ipc_telemetry_dropped_total"]) for item in runs
            ),
        }
        for mode, runs in results.items()
    }
    print(
        json.dumps({"configuration": config.model_dump(), "summary": summary}, indent=2)
    )


def run_condition(
    mode: Mode,
    config: ExperimentConfig,
    data_directory: Path,
    output_directory: Path,
) -> dict[str, float | int]:
    if mode == "headless":
        session = WorkerSession.create(config, data_directory, output_directory)
        started = time.perf_counter()
        while session.status == "running":
            session.advance()
        wall = time.perf_counter() - started
        return {
            "worker_loop_seconds": session.loop_seconds,
            "wall_seconds": wall,
            "browser_frames_dropped": 0,
            "ipc_telemetry_dropped_total": 0,
        }

    controller = PanelController(data_directory, output_directory)
    hub = TelemetryHub()
    client = hub.subscribe() if mode in {"connected", "slow"} else None
    controller.start_worker()
    controller.command("start", config)
    started = time.perf_counter()
    completed: dict[str, Any] = {}
    try:
        while True:
            event = controller.poll_latest()
            if event is not None:
                hub.publish(event)
                completed = event
            if mode == "connected" and client is not None:
                while not client.empty():
                    client.get_nowait()
            if completed.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.01)
    finally:
        controller.shutdown()
    return {
        "worker_loop_seconds": float(completed["worker_loop_seconds"]),
        "wall_seconds": time.perf_counter() - started,
        "browser_frames_dropped": hub.browser_frames_dropped,
        "ipc_telemetry_dropped_total": int(
            completed.get("ipc_telemetry_dropped_total", 0)
        ),
    }


if __name__ == "__main__":
    main()
