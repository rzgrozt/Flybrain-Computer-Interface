"""Isolated neural worker with bounded command and latest-value telemetry IPC."""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil

from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.panel.anatomy import ATLAS_DIRECTORY, AtlasJoin
from flybrain_interface.panel.models import ExperimentConfig, PathwayObservationConfig
from flybrain_interface.panel.pathway_observation import resolve_observation
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import SparseLIFSimulator
from flybrain_interface.simulation.trace import ChunkRecording

Command = dict[str, Any]
Telemetry = dict[str, Any]


class SignPolicyMetadata(Protocol):
    @property
    def name(self) -> str: ...


class PanelConnectome(Protocol):
    @property
    def neuron_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    @property
    def directory(self) -> Path: ...

    @property
    def sign_policy(self) -> SignPolicyMetadata: ...

    def accumulate_spikes(
        self,
        spiking_indices: Any,
        destination: np.ndarray[Any, np.dtype[np.float64]],
        *,
        amplitudes: Any | None = None,
    ) -> int: ...


def worker_main(
    commands: Queue[Command],
    telemetry: Queue[Telemetry],
    data_directory: str,
    output_directory: str,
) -> None:
    """Run until shutdown; never wait for a browser or telemetry consumer."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    session: WorkerSession | None = None
    dropped_total = publish_latest(telemetry, {"status": "idle", "kind": "status"}, 0)
    while True:
        if session is None or session.status != "running":
            try:
                command = commands.get(timeout=0.1)
            except queue.Empty:
                # A one-slot multiprocessing queue can transiently reject the
                # first transition frame while its feeder drains. Re-announce
                # paused state at the bounded telemetry rate so resume remains
                # available without making the simulation wait for a consumer.
                if session is not None and session.status == "paused":
                    if session.telemetry_due():
                        dropped_total = publish_latest(
                            telemetry,
                            session.snapshot("paused_heartbeat"),
                            dropped_total,
                        )
                continue
        else:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = {"action": "advance"}

        action = command.get("action")
        try:
            if action == "shutdown":
                if session is not None and session.status in {"running", "paused"}:
                    session.finish("stopped")
                dropped_total = publish_latest(
                    telemetry,
                    {"status": "stopped", "kind": "status"},
                    dropped_total,
                )
                return
            if action == "start":
                if session is not None and session.status in {"running", "paused"}:
                    raise ValueError("an experiment is already active")
                config = ExperimentConfig.model_validate(command["config"])
                session = WorkerSession.create(
                    config, Path(data_directory), Path(output_directory)
                )
                dropped_total = publish_latest(
                    telemetry, session.snapshot("started"), dropped_total
                )
            elif action == "pause":
                _require_session(session, "running")
                assert session is not None
                session.pause()
                dropped_total = publish_latest(
                    telemetry, session.snapshot("paused"), dropped_total
                )
            elif action == "resume":
                _require_session(session, "paused")
                assert session is not None
                session.resume()
                dropped_total = publish_latest(
                    telemetry, session.snapshot("resumed"), dropped_total
                )
            elif action == "reset":
                if session is not None:
                    session.finish("reset")
                session = None
                dropped_total = publish_latest(
                    telemetry,
                    {"kind": "status", "status": "idle", "reason": "reset"},
                    dropped_total,
                )
            elif action == "observe_pathway":
                if session is None or session.status not in {"running", "paused"}:
                    raise ValueError(
                        "pathway observation requires an active simulation"
                    )
                session.configure_pathway(command.get("observation"))
                dropped_total = publish_latest(
                    telemetry,
                    session.snapshot("pathway_selection_changed"),
                    dropped_total,
                )
            elif action == "advance":
                assert session is not None
                session.advance()
                if session.telemetry_due() or session.status == "completed":
                    dropped_total = publish_latest(
                        telemetry, session.snapshot("sample"), dropped_total
                    )
                if session.status == "completed":
                    session.finish("completed")
            else:
                raise ValueError(f"unsupported worker action: {action}")
        except Exception as error:
            active_duplicate_start = (
                action == "start"
                and session is not None
                and session.status in {"running", "paused"}
            )
            fatal = action == "advance" or (
                action == "start" and not active_duplicate_start
            )
            if session is not None and fatal:
                session.status = "failed"
                session.finish("failed", error=str(error))
            dropped_total = publish_latest(
                telemetry,
                {
                    "kind": "error",
                    "status": (
                        "failed" if session is None or fatal else session.status
                    ),
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                dropped_total,
            )


class WorkerSession:
    """Mutable experiment state owned only by the worker process."""

    def __init__(
        self,
        config: ExperimentConfig,
        simulator: SparseLIFSimulator,
        connectome: PanelConnectome,
        output_path: Path,
        dataset_report: dict[str, Any],
        atlas_join: AtlasJoin | None = None,
    ) -> None:
        self.config = config
        self.simulator = simulator
        self.connectome = connectome
        self.output_path = output_path
        self.dataset_report = dataset_report
        self.atlas_join = atlas_join
        self.experiment_id = output_path.name
        self.status = "running"
        self.started_at = datetime.now(UTC).isoformat()
        self.wall_started = time.perf_counter()
        self.paused_started: float | None = None
        self.total_paused_s = 0.0
        self.last_telemetry_wall = 0.0
        self.total_spikes = 0
        self.last_chunk_spikes = 0
        self.last_rates: dict[str, float] = {}
        self.last_voltage: list[float] = []
        self.last_drive: list[float] = []
        self.chunks = 0
        self.loop_seconds = 0.0
        self.last_activity_frame = _empty_activity_frame(config.chunk_duration_s)
        self.pathway_selection: dict[str, Any] | None = None
        self.last_pathway_measurement: dict[str, Any] | None = None
        if config.pathway_observation is not None:
            self.configure_pathway(config.pathway_observation.model_dump())

    @classmethod
    def create(
        cls, config: ExperimentConfig, data_directory: Path, output_directory: Path
    ) -> WorkerSession:
        graph = MemoryMappedConnectome.load(data_directory)
        _validate_indices(config, graph.neuron_count)
        populations = {
            population.name: tuple(population.neuron_indices)
            for population in config.populations
        }
        simulator = SparseLIFSimulator(
            graph,
            populations=populations,
            config=ShiuLIFConfig(),
            backend=config.backend,
            subnormal_drive_policy=config.subnormal_drive_policy,
        )
        simulator.prepare()
        experiment_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        output_path = output_directory / experiment_id
        output_path.mkdir(parents=True, exist_ok=False)
        report = _dataset_provenance(data_directory)
        atlas_join = AtlasJoin.load(ATLAS_DIRECTORY, data_directory)
        session = cls(
            config,
            simulator,
            graph,
            output_path,
            report,
            atlas_join=atlas_join,
        )
        session.write_manifest("running")
        return session

    def configure_pathway(self, observation: dict[str, Any] | None) -> None:
        """Change only the bounded trace watchlist at a chunk boundary."""
        # API commands carry server-resolved metadata; re-derive from their two
        # authoritative identifiers rather than trusting echoed index arrays.
        request = (
            PathwayObservationConfig.model_validate(
                {key: observation[key] for key in ("target_neuron_index", "path_index")}
            )
            if observation is not None
            else None
        )
        selection = resolve_observation(request) if request is not None else None
        self.pathway_selection = selection
        self.last_pathway_measurement = None

    def advance(self) -> None:
        remaining = self.config.duration_s - self.simulator.current_time_s
        if remaining <= 1e-12:
            self.status = "completed"
            return
        chunk_s = min(self.config.chunk_duration_s, remaining)
        # Floating arithmetic is snapped onto the configured neural grid.
        dt_s = self.simulator.config.dt_ms / 1000.0
        chunk_s = max(dt_s, round(chunk_s / dt_s) * dt_s)
        stimulus = self._chunk_stimulus(chunk_s)
        started = time.perf_counter()
        route_indices = (
            self.pathway_selection["neuron_indices"] if self.pathway_selection else []
        )
        watched = tuple(dict.fromkeys([*self.config.watch_indices, *route_indices]))
        visualized = list(
            dict.fromkeys([*self.config.visualization_indices, *route_indices])
        )
        result = self.simulator.advance_chunk(
            stimulus,
            duration_s=chunk_s,
            recording=ChunkRecording(
                watched_indices=watched,
                include_neuron_counts=bool(visualized),
                max_spike_events=0,
            ),
        )
        self.loop_seconds += time.perf_counter() - started
        self.chunks += 1
        self.last_chunk_spikes = result.total_spikes
        self.total_spikes += result.total_spikes
        self.last_rates = result.population_rates_hz
        self.last_activity_frame = build_activity_frame(
            visualized,
            result.neuron_spike_counts,
            result.duration_s,
            self.atlas_join,
        )
        if result.voltage_mv.size:
            # Preserve the legacy watchlist contract; extra route traces are separate.
            self.last_voltage = result.voltage_mv[
                -1, : len(self.config.watch_indices)
            ].tolist()
            self.last_drive = result.synaptic_drive_mv[
                -1, : len(self.config.watch_indices)
            ].tolist()
        self.last_pathway_measurement = None
        if self.pathway_selection is not None and result.voltage_mv.size:
            columns = [watched.index(index) for index in route_indices]
            voltages = [float(result.voltage_mv[-1, column]) for column in columns]
            drives = [float(result.synaptic_drive_mv[-1, column]) for column in columns]
            counts = result.neuron_spike_counts
            spikes = (
                [int(counts[index]) for index in route_indices]
                if counts is not None
                else None
            )
            self.last_pathway_measurement = {
                **self.pathway_selection,
                "source": "simulated_neural_measurement",
                "sampled_chunk": self.chunks,
                "simulated_time_s": self.simulator.current_time_s,
                "bin_duration_s": result.duration_s,
                "voltage_mv": voltages,
                "synaptic_drive_mv": drives,
                "spike_counts": spikes,
                "population_voltage_mean_mv": float(np.mean(voltages)),
                "population_drive_mean_mv": float(np.mean(drives)),
                "population_spikes": sum(spikes) if spikes is not None else None,
            }
        if self.simulator.current_time_s + 1e-12 >= self.config.duration_s:
            self.status = "completed"

    def _chunk_stimulus(self, chunk_s: float) -> DeterministicSpikeInput:
        start = self.simulator.current_time_s
        stop = start + chunk_s
        spec = self.config.stimulus
        interval_s = spec.interval_ms / 1000.0
        offset = (max(start, spec.start_s) - spec.start_s) / interval_s
        first_number = max(0, int(np.ceil(offset - 1e-12)))
        times: list[float] = []
        indices: list[int] = []
        number = first_number
        while True:
            absolute = spec.start_s + number * interval_s
            if absolute >= min(stop, spec.stop_s) - 1e-12:
                break
            if absolute >= start - 1e-12:
                dt_s = self.simulator.config.dt_ms / 1000.0
                relative = round((absolute - start) / dt_s) * dt_s
                for neuron in spec.neuron_indices:
                    indices.append(neuron)
                    times.append(relative)
            number += 1
        return DeterministicSpikeInput(
            neuron_indices=tuple(indices),
            times_s=tuple(times),
            amplitude_mv=spec.amplitude_mv,
        )

    def telemetry_due(self) -> bool:
        now = time.perf_counter()
        if now - self.last_telemetry_wall >= 1.0 / self.config.telemetry_hz:
            self.last_telemetry_wall = now
            return True
        return False

    def pause(self) -> None:
        self.status = "paused"
        self.paused_started = time.perf_counter()

    def resume(self) -> None:
        if self.paused_started is not None:
            self.total_paused_s += time.perf_counter() - self.paused_started
        self.paused_started = None
        self.status = "running"

    def snapshot(self, reason: str) -> Telemetry:
        now = time.perf_counter()
        wall_elapsed_s = max(0.0, now - self.wall_started)
        current_pause_s = (
            now - self.paused_started if self.paused_started is not None else 0.0
        )
        active_wall_s = max(0.0, wall_elapsed_s - self.total_paused_s - current_pause_s)
        return {
            "kind": "telemetry",
            "reason": reason,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "simulated_time_s": self.simulator.current_time_s,
            "duration_s": self.config.duration_s,
            "wall_elapsed_s": wall_elapsed_s,
            "active_wall_time_s": active_wall_s,
            "speed_ratio": (
                self.simulator.current_time_s / active_wall_s if active_wall_s else 0.0
            ),
            "process_rss_bytes": _rss_bytes(),
            "pending_delayed_events": self.simulator.pending_delayed_events,
            "total_spikes": self.total_spikes,
            "chunk_spikes": self.last_chunk_spikes,
            "population_rates_hz": self.last_rates,
            "watch_indices": self.config.watch_indices,
            "voltage_mv": self.last_voltage,
            "synaptic_drive_mv": self.last_drive,
            "worker_loop_seconds": self.loop_seconds,
            "chunks": self.chunks,
            "visited_edges": self.simulator.visited_edges,
            "subnormal_drive_policy": self.config.subnormal_drive_policy,
            "brain_activity": self.last_activity_frame,
            "pathway_selection": self.pathway_selection,
            "pathway_measurement": self.last_pathway_measurement,
        }

    def finish(self, status: str, *, error: str | None = None) -> None:
        self.status = status
        self.write_manifest(status, error=error)

    def write_manifest(self, status: str, *, error: str | None = None) -> None:
        cfg = self.simulator.config
        manifest = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "status": status,
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": str(self.connectome.directory),
                "name": self.dataset_report.get("dataset", "male-cns:v1.0"),
                "neuron_count": self.connectome.neuron_count,
                "edge_count": self.connectome.edge_count,
                "normalized_sha256": self.dataset_report.get("output_sha256", {}),
                "source_manifest": self.dataset_report.get("source_manifest", {}),
                "normalized_manifest": self.dataset_report.get(
                    "normalized_manifest", {}
                ),
            },
            "sign_policy": self.connectome.sign_policy.name,
            "neural_parameters": asdict(cfg),
            "configuration": self.config.model_dump(mode="json"),
            "recording": {
                "mode": "bounded latest-value telemetry",
                "watchlist_limit": 32,
                "telemetry_hz": self.config.telemetry_hz,
                "spike_event_history_limit": 0,
                "visualization_neuron_limit": 4096,
                "visualization_signal": "emitted simulated spike count per chunk",
                "visualization_normalization_reference_rate_hz": 50.0,
            },
            "numerical_approximation": (
                "none; preserve exact IEEE-754 subnormal decay"
                if self.config.subnormal_drive_policy == "preserve"
                else "drive values below the normal float64 range are set to zero"
            ),
            "result": {
                "simulated_time_s": self.simulator.current_time_s,
                "total_spikes": self.total_spikes,
                "pending_delayed_events": self.simulator.pending_delayed_events,
                "worker_loop_seconds": self.loop_seconds,
            },
            "software": {
                "python": sys.version.split()[0],
                "flybrain-interface": version("flybrain-interface"),
                "numpy": version("numpy"),
                "numba": version("numba"),
                "fastapi": version("fastapi"),
            },
            "error": error,
        }
        temporary = self.output_path / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.output_path / "manifest.json")


def _validate_indices(config: ExperimentConfig, neuron_count: int) -> None:
    groups = [
        config.watch_indices,
        config.visualization_indices,
        config.stimulus.neuron_indices,
    ]
    groups.extend(population.neuron_indices for population in config.populations)
    if any(index >= neuron_count for group in groups for index in group):
        raise ValueError(f"neuron index must be below {neuron_count}")
    dt_ms = ShiuLIFConfig().dt_ms
    for value, label in (
        (config.chunk_duration_s * 1000.0, "chunk_duration_s"),
        (config.stimulus.start_s * 1000.0, "stimulus.start_s"),
        (config.stimulus.stop_s * 1000.0, "stimulus.stop_s"),
        (config.stimulus.interval_ms, "stimulus.interval_ms"),
        (config.duration_s * 1000.0, "duration_s"),
    ):
        if not np.isclose(value / dt_ms, round(value / dt_ms), atol=1e-9):
            raise ValueError(f"{label} must align to the {dt_ms} ms neural grid")


def _dataset_provenance(data_directory: Path) -> dict[str, Any]:
    report_path = data_directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    manifest_directory = Path(__file__).parents[3] / "data" / "manifests"
    normalized_path = manifest_directory / "malecns-v1.0-normalized-v2.json"
    source_path = manifest_directory / "malecns-v1.0.json"
    if normalized_path.is_file():
        report["normalized_manifest"] = json.loads(normalized_path.read_text())
    if source_path.is_file():
        source = json.loads(source_path.read_text())
        report["source_manifest"] = {
            key: source[key]
            for key in (
                "schema_version",
                "dataset",
                "dataset_uuid",
                "release_date",
                "license",
                "landing_page",
                "sources",
            )
        }
    return report


def _require_session(session: WorkerSession | None, expected: str) -> None:
    if session is None or session.status != expected:
        actual = "idle" if session is None else session.status
        raise ValueError(f"action requires {expected} state; worker is {actual}")


def publish_latest(
    queue_: Queue[Telemetry], event: Telemetry, dropped_total: int = 0
) -> int:
    """Never block: replace obsolete telemetry and count replacement attempts."""

    event = dict(event)
    try:
        event["ipc_telemetry_dropped_total"] = dropped_total
        queue_.put_nowait(event)
        return dropped_total
    except queue.Full:
        pass
    try:
        queue_.get_nowait()
        dropped_total += 1
    except queue.Empty:
        pass
    event["ipc_telemetry_dropped_total"] = dropped_total
    try:
        queue_.put_nowait(event)
    except queue.Full:
        dropped_total += 1
    return dropped_total


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def build_activity_frame(
    selected_indices: list[int],
    counts: np.ndarray[Any, np.dtype[np.int64]] | None,
    bin_duration_s: float,
    atlas_join: AtlasJoin | None,
) -> dict[str, Any]:
    """Aggregate one bounded chunk; never retain or interpolate spike events."""

    frame = _empty_activity_frame(bin_duration_s)
    frame["selected_neuron_count"] = len(selected_indices)
    if counts is None or not selected_indices:
        return frame
    selected = np.asarray(selected_indices, dtype=np.int64)
    active = selected[counts[selected] > 0]
    values: list[list[Any]] = []
    without_visible_soma = 0
    for index in active:
        neuron_index = int(index)
        spike_count = int(counts[neuron_index])
        if atlas_join is None:
            body_id = str(neuron_index)
            visible = False
        else:
            body_id = str(int(atlas_join.body_ids_by_index[neuron_index]))
            visible = int(atlas_join.visible_atlas_row_by_index[neuron_index]) >= 0
        if not visible:
            without_visible_soma += 1
            continue
        rate_hz = spike_count / bin_duration_s
        values.append([neuron_index, body_id, min(1.0, rate_hz / 50.0), spike_count])
    frame["values"] = values
    frame["active_selected_count"] = int(active.size)
    frame["active_without_visible_soma_count"] = without_visible_soma
    return frame


def _empty_activity_frame(bin_duration_s: float) -> dict[str, Any]:
    return {
        "signal": "emitted_simulated_spike_count",
        "units": "spikes per neuron per simulation-time bin",
        "bin_duration_s": bin_duration_s,
        "normalization": {
            "method": "linear_rate_clamped_0_1",
            "reference_rate_hz": 50.0,
            "meaning": "display brightness only; not voltage or synaptic current",
        },
        "values": [],
        "selected_neuron_count": 0,
        "active_selected_count": 0,
        "active_without_visible_soma_count": 0,
    }
