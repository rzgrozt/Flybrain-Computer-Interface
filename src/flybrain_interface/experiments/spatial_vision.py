"""Validate measured retinotopy with deterministic moving-screen bars."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import load_optic_columns
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.experiments.retinotopy import (
    infer_r1r6_columns,
    summarize_official_source_consistency,
)
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.sensory.vision import (
    RetinotopicScreenEncoding,
    VirtualScreenConfig,
    encode_screen_sequence,
    load_measured_column_directions,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-vision-v1.json"
Axis = Literal["horizontal", "vertical"]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial-vision config schema")
    return payload


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    retinotopy_config_path = ROOT / str(config["retinotopy_config"])
    workbook_path = ROOT / str(config["optic_workbook"])
    eye_map_manifest_path = ROOT / str(config["eye_map_manifest"])
    directions_path = ROOT / str(config["measured_directions"])

    eye_map_lock = _verify_measured_directions(
        directions_path,
        eye_map_manifest_path,
    )
    directions = load_measured_column_directions(directions_path)
    retinotopy_config = load_retinotopy_config(retinotopy_config_path)
    columns = load_optic_columns(workbook_path)
    graph = MemoryMappedConnectome.load(data_directory)
    source_consistency = summarize_official_source_consistency(graph, columns)
    assignments = infer_r1r6_columns(graph, columns, retinotopy_config)

    allowed_statuses = {str(value) for value in config["neuron_assignment_statuses"]}
    occupancy = Counter(
        assignment.assigned_column
        for assignment in assignments
        if assignment.status in allowed_statuses
        and assignment.assigned_column is not None
    )

    screen_spec = config["screen"]
    screen = VirtualScreenConfig(
        horizontal_fov_deg=float(screen_spec["horizontal_fov_deg"]),
        vertical_fov_deg=float(screen_spec["vertical_fov_deg"]),
    )
    positions = tuple(float(value) for value in screen_spec["positions"])
    if len(positions) < 3 or any(
        not 0.0 < position < 1.0 for position in positions
    ):
        raise ValueError(
            "screen positions must contain at least three values in (0, 1)"
        )
    frame_rate_hz = float(config["timing"]["frame_rate_hz"])

    sweeps: dict[str, dict[str, Any]] = {}
    for axis in ("horizontal", "vertical"):
        frames = _bar_frames(axis, positions, screen_spec)
        encoding = encode_screen_sequence(
            frames,
            directions,
            screen,
            frame_rate_hz=frame_rate_hz,
        )
        sweeps[axis] = _sweep_metrics(
            axis,
            positions,
            encoding,
            occupancy,
            background=float(screen_spec["background_intensity"]),
        )

    visible_columns = len(
        {
            projection.column
            for projection in encode_screen_sequence(
                _bar_frames("horizontal", (positions[0],), screen_spec),
                directions,
                screen,
                frame_rate_hz=frame_rate_hz,
            ).projections
        }
    )
    visible_r1r6 = sum(
        occupancy.get(column, 0)
        for column in sweeps["horizontal"]["visible_columns"]
    )
    gates = _evaluate_gates(
        config,
        visible_columns=visible_columns,
        visible_r1r6=visible_r1r6,
        sweeps=sweeps,
    )

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Geometry-only validation of a perspective virtual screen sampled by "
            "measured optic-column viewing directions. No neural drive or motor "
            "meaning is implied."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "retinotopy_config_sha256": file_sha256(retinotopy_config_path),
        "eye_map_lock": eye_map_lock,
        "official_source_consistency": source_consistency,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "measured_direction_count": len(directions),
        "visible_measured_column_count": visible_columns,
        "visible_high_confidence_r1r6_count": visible_r1r6,
        "sweeps": sweeps,
        "gates": gates,
    }


def _bar_frames(
    axis: Axis,
    positions: tuple[float, ...],
    screen_spec: dict[str, Any],
) -> np.ndarray:
    width = int(screen_spec["width_px"])
    height = int(screen_spec["height_px"])
    background = float(screen_spec["background_intensity"])
    foreground = float(screen_spec["bar_intensity"])
    bar_width = float(screen_spec["bar_width_fraction"])
    if width <= 1 or height <= 1:
        raise ValueError("screen dimensions must be greater than one pixel")
    if not 0.0 <= background <= 1.0 or not 0.0 <= foreground <= 1.0:
        raise ValueError("screen intensities must be normalized to [0, 1]")
    if not 0.0 < bar_width < 1.0:
        raise ValueError("bar_width_fraction must be in (0, 1)")

    frames = np.full(
        (len(positions), height, width),
        background,
        dtype=np.float64,
    )
    extent = width if axis == "horizontal" else height
    for frame_index, position in enumerate(positions):
        low = max(0, int(round((position - bar_width / 2.0) * (extent - 1))))
        high = min(
            extent,
            int(round((position + bar_width / 2.0) * (extent - 1))) + 1,
        )
        if axis == "horizontal":
            frames[frame_index, :, low:high] = foreground
        else:
            frames[frame_index, low:high, :] = foreground
    return frames


def _sweep_metrics(
    axis: Axis,
    positions: tuple[float, ...],
    encoding: RetinotopicScreenEncoding,
    occupancy: Counter[str],
    *,
    background: float,
) -> dict[str, Any]:
    coordinates = np.asarray(
        [
            projection.u if axis == "horizontal" else projection.v
            for projection in encoding.projections
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(coordinates)):
        raise RuntimeError("visible screen projection contains non-finite coordinates")
    neuron_weights = np.asarray(
        [occupancy.get(column, 0) for column in encoding.columns],
        dtype=np.float64,
    )

    column_centroids: list[float] = []
    r1r6_centroids: list[float] = []
    frames: list[dict[str, Any]] = []
    for position, intensity in zip(
        positions,
        encoding.intensity_by_frame,
        strict=True,
    ):
        contrast = np.maximum(0.0, intensity - background)
        column_total = float(np.sum(contrast))
        neuron_contrast = contrast * neuron_weights
        neuron_total = float(np.sum(neuron_contrast))
        if column_total <= 0.0 or neuron_total <= 0.0:
            raise RuntimeError(f"{axis} sweep contains a bar with no sampled response")
        column_centroid = float(np.sum(coordinates * contrast) / column_total)
        r1r6_centroid = float(
            np.sum(coordinates * neuron_contrast) / neuron_total
        )
        column_centroids.append(column_centroid)
        r1r6_centroids.append(r1r6_centroid)
        active = contrast > 1e-12
        frames.append(
            {
                "bar_position": position,
                "active_column_count": int(np.count_nonzero(active)),
                "active_high_confidence_r1r6_count": int(
                    np.sum(neuron_weights[active])
                ),
                "column_centroid": column_centroid,
                "r1r6_centroid": r1r6_centroid,
            }
        )

    return {
        "axis": axis,
        "visible_columns": encoding.columns,
        "frame_sha256": encoding.frame_sha256,
        "frames": frames,
        "column_centroid_correlation": _correlation(positions, column_centroids),
        "r1r6_centroid_correlation": _correlation(positions, r1r6_centroids),
        "column_centroid_monotonic": _strictly_increasing(column_centroids),
        "r1r6_centroid_monotonic": _strictly_increasing(r1r6_centroids),
    }


def _evaluate_gates(
    config: dict[str, Any],
    *,
    visible_columns: int,
    visible_r1r6: int,
    sweeps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    specification = config["gates"]
    minimum_column_correlation = float(
        specification["minimum_column_centroid_correlation"]
    )
    minimum_r1r6_correlation = float(
        specification["minimum_r1r6_centroid_correlation"]
    )
    checks = {
        "visible_measured_columns": visible_columns
        >= int(specification["minimum_visible_measured_columns"]),
        "visible_high_confidence_r1r6": visible_r1r6
        >= int(specification["minimum_visible_high_confidence_r1r6"]),
        "horizontal_column_correlation": (
            sweeps["horizontal"]["column_centroid_correlation"]
            >= minimum_column_correlation
        ),
        "vertical_column_correlation": (
            sweeps["vertical"]["column_centroid_correlation"]
            >= minimum_column_correlation
        ),
        "horizontal_r1r6_correlation": (
            sweeps["horizontal"]["r1r6_centroid_correlation"]
            >= minimum_r1r6_correlation
        ),
        "vertical_r1r6_correlation": (
            sweeps["vertical"]["r1r6_centroid_correlation"]
            >= minimum_r1r6_correlation
        ),
    }
    if bool(specification["require_monotonic_centroid"]):
        checks["horizontal_column_monotonic"] = bool(
            sweeps["horizontal"]["column_centroid_monotonic"]
        )
        checks["vertical_column_monotonic"] = bool(
            sweeps["vertical"]["column_centroid_monotonic"]
        )
        checks["horizontal_r1r6_monotonic"] = bool(
            sweeps["horizontal"]["r1r6_centroid_monotonic"]
        )
        checks["vertical_r1r6_monotonic"] = bool(
            sweeps["vertical"]["r1r6_centroid_monotonic"]
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": specification,
    }


def _verify_measured_directions(
    path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    derived = manifest.get("derived_output")
    if not isinstance(derived, dict):
        raise ValueError("eye-map manifest is missing derived_output lock")
    digest = file_sha256(path)
    expected = str(derived["sha256"])
    if digest != expected:
        raise ValueError("measured-direction CSV SHA-256 mismatch")
    return {
        "path": str(path),
        "record_count": int(derived["record_count"]),
        "sha256": digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _correlation(expected: tuple[float, ...], actual: list[float]) -> float:
    if len(expected) != len(actual) or len(actual) < 2:
        raise ValueError(
            "correlation requires aligned vectors with at least two values"
        )
    value = float(np.corrcoef(expected, actual)[0, 1])
    if not np.isfinite(value):
        raise RuntimeError("spatial centroid correlation is non-finite")
    return value


def _strictly_increasing(values: list[float]) -> bool:
    return all(after > before for before, after in zip(values, values[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = run_validation(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "visible_measured_column_count": result[
                    "visible_measured_column_count"
                ],
                "visible_high_confidence_r1r6_count": result[
                    "visible_high_confidence_r1r6_count"
                ],
                "gates": result["gates"],
                "sweeps": {
                    axis: {
                        key: value
                        for key, value in sweep.items()
                        if key != "visible_columns"
                    }
                    for axis, sweep in result["sweeps"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
