"""Replay bounded pixels through canonical vision and a fixed DN rate decoder.

This is an open-loop signal-availability baseline, not a trained policy or a
desktop-control loop. Frame pixels are the sole external sensory input.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.validate import validate_processed_dataset
from flybrain_interface.contracts import NeuralReadout
from flybrain_interface.experiments.spatial_graded_lamina import (
    _make_lamina_drive,
    _top_downstream_targets,
)
from flybrain_interface.experiments.spatial_motor_pathway import (
    VisualSetup,
    _setup_visual_system,
)
from flybrain_interface.experiments.spatial_motor_readout import (
    resolve_motor_populations,
)
from flybrain_interface.experiments.spatial_neural_vision import (
    _source_column_intensities,
)
from flybrain_interface.motor.fixed import FixedPopulationMotorDecoder
from flybrain_interface.sensory.drive import DeterministicProjectedDriveInput
from flybrain_interface.sensory.vision import (
    encode_screen_sequence,
    simulate_graded_early_vision_channels,
)
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import ChunkRecording

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs/pixel-fixed-motor-v1.json"


def normalize_frames(frames: np.ndarray, limits: dict[str, int]) -> np.ndarray:
    """Accept bounded grayscale/RGB pixels; never extract objects or coordinates."""
    if frames.ndim not in (3, 4) or any(size <= 0 for size in frames.shape):
        raise ValueError("frames must have shape (time, height, width[, 3])")
    if frames.ndim == 4 and frames.shape[3] != 3:
        raise ValueError("color frames must have exactly three RGB channels")
    if (
        frames.shape[0] > limits["max_frames"]
        or frames.shape[1] * frames.shape[2] > limits["max_pixels_per_frame"]
    ):
        raise ValueError("pixel sequence exceeds configured frame/pixel limits")
    if frames.dtype == np.uint8:
        return np.asarray(frames, dtype=np.float64) / 255.0
    if not np.issubdtype(frames.dtype, np.floating):
        raise ValueError("pixels must be uint8 or normalized floating point")
    if not np.isfinite(frames).all() or np.any((frames < 0) | (frames > 1)):
        raise ValueError("floating point pixels must be finite and in [0, 1]")
    return np.asarray(frames, dtype=np.float64)


def demo_frames() -> np.ndarray:
    """A fixed 2D dark square; geometry stays inside this stimulus generator."""
    frames = np.full((12, 144, 256), 0.5, dtype=np.float64)
    frames[3:9, 30:70, 150:210] = 0.0
    return frames


def slice_drive(
    drive: DeterministicProjectedDriveInput | None, start: int, stop: int
) -> DeterministicProjectedDriveInput | None:
    """Retain already-delayed drive across state-preserving motor chunks."""
    if drive is None:
        return None
    return DeterministicProjectedDriveInput(
        channels=tuple(
            replace(channel, amplitudes_mv=channel.amplitudes_mv[start:stop])
            for channel in drive.channels
        ),
        dt_ms=drive.dt_ms,
    )


def run_sequence(
    setup: VisualSetup,
    simulator: SparseLIFSimulator,
    frames: np.ndarray,
    decoder: FixedPopulationMotorDecoder,
    *,
    chunk_ms: float,
    disconnected: bool = False,
) -> dict[str, Any]:
    """Reset once per condition; retain recurrent state between motor ticks."""
    frame_rate = setup.graded_config.frame_rate_hz
    exact_steps = frames.shape[0] / frame_rate * 1000 / setup.lif_config.dt_ms
    exact_chunk = chunk_ms / setup.lif_config.dt_ms
    if (
        not math.isfinite(exact_chunk)
        or exact_chunk < 1
        or not math.isclose(exact_chunk, round(exact_chunk), rel_tol=0, abs_tol=1e-9)
        or not math.isclose(exact_steps, round(exact_steps), rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("sequence duration and motor chunks must align to neural dt")
    encoding = encode_screen_sequence(
        frames, setup.directions, setup.screen, frame_rate_hz=frame_rate
    )
    intensities = _source_column_intensities(
        encoding,
        setup.source_columns,
        background=setup.graded_config.background_luminance,
    )
    graded = simulate_graded_early_vision_channels(intensities, setup.graded_config)
    projected = _make_lamina_drive(
        graded, setup.cartridges, setup.projections, setup.base, setup.lif_config
    )
    drive = None if disconnected else projected
    downstream = _top_downstream_targets(
        setup.projections, set(setup.clamped), limit=16
    )
    watch = tuple(
        sorted(
            set(downstream).union(
                index for indices in simulator.populations.values() for index in indices
            )
        )
    )
    simulator.reset()
    ticks: list[dict[str, Any]] = []
    peak_downstream_mv = 0.0
    peak_motor_mv = 0.0
    total_spikes = 0
    clamped_spikes = 0
    started = perf_counter()
    for start in range(0, round(exact_steps), round(exact_chunk)):
        stop = min(start + round(exact_chunk), round(exact_steps))
        result = simulator.advance_chunk(
            duration_s=(stop - start) * setup.lif_config.dt_ms / 1000,
            recording=ChunkRecording(watched_indices=watch, include_neuron_counts=True),
            projected_drive=slice_drive(drive, start, stop),
        )
        assert result.neuron_spike_counts is not None
        counts = result.neuron_spike_counts
        readout = NeuralReadout(
            duration_s=result.duration_s,
            neuron_spike_counts=(),
            population_rates_hz=result.population_rates_hz,
        )
        command = decoder.decode(readout)
        deltas = result.voltage_mv - simulator.config.resting_mv
        for position, index in enumerate(watch):
            peak = float(np.max(np.abs(deltas[:, position])))
            if index in downstream:
                peak_downstream_mv = max(peak_downstream_mv, peak)
            if any(index in indices for indices in simulator.populations.values()):
                peak_motor_mv = max(peak_motor_mv, peak)
        total_spikes += result.total_spikes
        clamped_spikes += int(counts[np.asarray(setup.clamped, dtype=np.int64)].sum())
        if (
            not np.isfinite(simulator.voltage_mv).all()
            or not np.isfinite(simulator.synaptic_drive_mv).all()
        ):
            raise RuntimeError("nonfinite neural state")
        ticks.append(
            {
                "start_s": start * setup.lif_config.dt_ms / 1000,
                "end_s": stop * setup.lif_config.dt_ms / 1000,
                "population_rates_hz": result.population_rates_hz,
                "population_voltage_delta_mv": result.population_voltage_delta_mv,
                "population_synaptic_drive_mv": result.population_synaptic_drive_mv,
                "motor_command": asdict(command),
            }
        )
    elapsed = perf_counter() - started
    duration = exact_steps * setup.lif_config.dt_ms / 1000
    return {
        "frame_sha256": encoding.frame_sha256,
        "projected_channels": 0 if drive is None else len(drive.channels),
        "max_abs_release_delta": float(
            np.max(np.abs(graded.photoreceptor.histamine_release_delta))
        ),
        "total_network_spikes": total_spikes,
        "clamped_visual_spikes": clamped_spikes,
        "peak_downstream_voltage_delta_mv": peak_downstream_mv,
        "peak_motor_voltage_delta_mv": peak_motor_mv,
        "motor_nonzero": any(
            tick["motor_command"]["delta_x"] != 0
            or tick["motor_command"]["delta_y"] != 0
            for tick in ticks
        ),
        "duration_s": duration,
        "neural_execution_wall_s": elapsed,
        "simulated_to_execution_wall_ratio": duration / elapsed,
        "ticks": ticks,
    }


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
    frames_path: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported pixel fixed motor config schema")
    if frames_path is not None and frames_path.suffix != ".npy":
        raise ValueError("--frames requires a .npy pixel sequence")
    frames = normalize_frames(
        demo_frames()
        if frames_path is None
        else np.load(frames_path, mmap_mode="r", allow_pickle=False),
        config["limits"],
    )
    verified_data = validate_processed_dataset(
        ROOT / config["normalized_data_manifest"], data_directory
    )
    base_path = ROOT / config["base_config"]
    setup = _setup_visual_system(data_directory, base_path)
    motor_path = ROOT / config["motor_config"]
    populations = resolve_motor_populations(
        setup.graph.catalog.table, json.loads(motor_path.read_text())
    )
    decoder = FixedPopulationMotorDecoder(**config["decoder"])
    mapping = {population.key: population.neuron_indices for population in populations}
    required = {
        decoder.left_population,
        decoder.right_population,
        decoder.up_population,
        decoder.down_population,
    }
    if not required <= mapping.keys() or set(setup.clamped).intersection(
        index for indices in mapping.values() for index in indices
    ):
        raise ValueError(
            "motor mapping is missing populations or overlaps visual clamps"
        )
    if (
        not math.isfinite(decoder.gain)
        or decoder.gain < 0
        or not (math.isfinite(decoder.max_delta) and decoder.max_delta > 0)
    ):
        raise ValueError(
            "fixed motor gain/limit must be finite and nonnegative/positive"
        )
    simulator = SparseLIFSimulator(
        setup.graph,
        populations=mapping,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, setup.base["runtime"]["backend"]),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy, setup.base["runtime"]["subnormal_drive_policy"]
        ),
    )
    simulator.prepare()
    results = {}
    for name, pixels, disconnected in (
        ("pixels", frames, False),
        (
            "blank",
            np.full_like(frames, setup.graded_config.background_luminance),
            False,
        ),
        ("vision_disconnected", frames, True),
    ):
        results[name] = run_sequence(
            setup,
            simulator,
            pixels,
            decoder,
            chunk_ms=float(config["motor_chunk_ms"]),
            disconnected=disconnected,
        )
    paths = {
        "experiment_config": config_path,
        "base_config": base_path,
        "motor_config": motor_path,
        "normalized_data_manifest": ROOT / config["normalized_data_manifest"],
        "retinotopy_config": ROOT / setup.base["retinotopy_config"],
        "optic_workbook": ROOT / setup.base["optic_workbook"],
        "eye_map_manifest": ROOT / setup.base["eye_map_manifest"],
        "measured_directions": ROOT / setup.base["measured_directions"],
        "experiment_source": Path(__file__),
        "retinal_encoder_source": ROOT
        / "src/flybrain_interface/sensory/vision/retinotopic.py",
        "graded_vision_source": ROOT
        / "src/flybrain_interface/sensory/vision/graded.py",
        "runtime_source": ROOT / "src/flybrain_interface/simulation/runtime.py",
        "kernel_source": ROOT / "src/flybrain_interface/simulation/kernels.py",
        "decoder_source": ROOT / "src/flybrain_interface/motor/fixed.py",
    }
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "scope": "open-loop pixel replay; no learning or target-seeking claim",
        "config": config,
        "base_config": setup.base,
        "lif_config": asdict(setup.lif_config),
        "source_hashes": {name: file_sha256(path) for name, path in paths.items()},
        "input": {
            "source": "built-in 2D square"
            if frames_path is None
            else str(frames_path.resolve()),
            "shape": list(frames.shape),
            "file_sha256": None if frames_path is None else file_sha256(frames_path),
        },
        "dataset": {
            "directory": str(data_directory.resolve()),
            "verified_files": verified_data.verified_files,
            "neurons": setup.graph.neuron_count,
            "edges": setup.graph.edge_count,
            "sign_policy": setup.graph.sign_policy.name,
        },
        "motor_populations": [asdict(population) for population in populations],
        "decoder": asdict(decoder),
        "runtime": {
            "backend": simulator.backend,
            "subnormal_drive_policy": simulator.subnormal_drive_policy,
            "graded_relays_enabled": False,
            "plasticity_enabled": False,
            "os_actions_enabled": False,
        },
        "versions": {
            name: version(name)
            for name in ("flybrain-interface", "numpy", "numba", "scipy", "brian2")
        },
        "seed": None,
        "stochastic_operations": False,
        "conditions": results,
        "integrity_checks": {
            "no_clamped_spikes": all(
                r["clamped_visual_spikes"] == 0 for r in results.values()
            ),
            "blank_silent": results["blank"]["total_network_spikes"] == 0,
            "disconnected_silent": results["vision_disconnected"][
                "total_network_spikes"
            ]
            == 0,
        },
        "signal_checks": {
            "pixels_reach_downstream": results["pixels"][
                "peak_downstream_voltage_delta_mv"
            ]
            > 0.01,
            "pixels_reach_motor": results["pixels"]["peak_motor_voltage_delta_mv"]
            > 0.001,
            "fixed_decoder_moves": results["pixels"]["motor_nonzero"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory", type=Path, default=ROOT / "data/processed/malecns-v1.0"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=Path, help="bounded .npy pixel sequence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_validation(args.data_directory, args.config, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "integrity_checks": result["integrity_checks"],
                "signal_checks": result["signal_checks"],
            },
            indent=2,
        )
    )
    if not all(result["integrity_checks"].values()):
        parser.exit(1, "Pixel baseline integrity checks failed; see output artifact.\n")


if __name__ == "__main__":
    main()
