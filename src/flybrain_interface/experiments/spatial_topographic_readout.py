"""Decode unseen position from anatomically backprojected subthreshold state."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import load_optic_columns
from flybrain_interface.connectome_data.runtime import (
    MemoryMappedConnectome,
    SignedSourceProjection,
)
from flybrain_interface.experiments.retinotopy import infer_r1r6_columns
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    _lamina_projections,
    _top_downstream_targets,
    _validate_lamina_signs,
    resolve_cartridge_sources,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    load_config as load_base_config,
)
from flybrain_interface.experiments.spatial_neural_vision import (
    _source_groups,
    _verify_measured_directions,
)
from flybrain_interface.experiments.spatial_subthreshold_readout import (
    _graded_config,
    _prediction_metrics,
    _run_position,
    _validate_positions,
    fit_ridge,
    predict_ridge,
)
from flybrain_interface.sensory.vision import (
    VirtualScreenConfig,
    load_measured_column_directions,
    project_directions_to_screen,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-topographic-readout-v1.json"
Axis = Literal["horizontal", "vertical"]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AnatomicalBackprojection:
    """Map watched downstream neurons back onto measured visual columns."""

    target_indices: tuple[int, ...]
    columns: tuple[str, ...]
    u: FloatArray
    v: FloatArray
    target_to_column: FloatArray
    target_contact_weight: FloatArray

    def __post_init__(self) -> None:
        target_count = len(self.target_indices)
        column_count = len(self.columns)
        if self.u.shape != (column_count,) or self.v.shape != (column_count,):
            raise ValueError("backprojection coordinates must align with columns")
        if self.target_to_column.shape != (target_count, column_count):
            raise ValueError("backprojection matrix shape is inconsistent")
        if self.target_contact_weight.shape != (target_count,):
            raise ValueError("target contact weights must align with targets")
        if not (
            np.isfinite(self.u).all()
            and np.isfinite(self.v).all()
            and np.isfinite(self.target_to_column).all()
            and np.isfinite(self.target_contact_weight).all()
        ):
            raise ValueError("backprojection arrays must be finite")
        if np.any(self.target_to_column < 0.0):
            raise ValueError("backprojection weights cannot be negative")
        if np.any(self.target_contact_weight <= 0.0):
            raise ValueError("every watched target must receive anatomical input")
        if not np.allclose(
            np.sum(self.target_to_column, axis=1),
            np.ones(target_count, dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("each target backprojection row must sum to one")

    def coordinates(self, axis: Axis) -> FloatArray:
        return self.u if axis == "horizontal" else self.v


@dataclass(frozen=True, slots=True)
class AffineModel:
    slope: float
    intercept: float


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial-topographic config schema")
    return payload


def build_anatomical_backprojection(
    projections: dict[tuple[str, str], SignedSourceProjection],
    watched_targets: tuple[int, ...],
    column_coordinates: dict[str, tuple[float, float]],
) -> AnatomicalBackprojection:
    """Aggregate L1/L2/L3 contacts into a normalized target-to-column map."""

    columns = tuple(sorted(column_coordinates))
    if not columns:
        raise ValueError("backprojection requires measured visible columns")
    if not watched_targets:
        raise ValueError("backprojection requires watched downstream targets")
    column_position = {column: index for index, column in enumerate(columns)}
    target_position = {
        target: index for index, target in enumerate(watched_targets)
    }
    weights = np.zeros(
        (len(watched_targets), len(columns)),
        dtype=np.float64,
    )
    for (_, column), projection in projections.items():
        column_index = column_position.get(column)
        if column_index is None:
            continue
        for target, weight in zip(
            projection.target_indices,
            projection.signed_contact_weights,
            strict=True,
        ):
            target_index = target_position.get(int(target))
            if target_index is not None:
                weights[target_index, column_index] += abs(float(weight))

    totals = np.sum(weights, axis=1)
    if np.any(totals <= 0.0):
        missing = [
            watched_targets[index]
            for index in np.flatnonzero(totals <= 0.0)
        ]
        raise RuntimeError(
            f"watched targets lack lamina backprojection: {missing[:10]}"
        )
    normalized = weights / totals[:, None]
    u = np.asarray(
        [column_coordinates[column][0] for column in columns],
        dtype=np.float64,
    )
    v = np.asarray(
        [column_coordinates[column][1] for column in columns],
        dtype=np.float64,
    )
    return AnatomicalBackprojection(
        target_indices=watched_targets,
        columns=columns,
        u=u,
        v=v,
        target_to_column=normalized,
        target_contact_weight=totals,
    )


def backproject_response(
    response: np.ndarray,
    mapping: AnatomicalBackprojection,
    *,
    epsilon: float,
) -> FloatArray:
    """Return normalized visual-column activity from downstream response magnitude."""

    values = np.asarray(response, dtype=np.float64)
    if values.shape != (len(mapping.target_indices),):
        raise ValueError("response must align with backprojection targets")
    if not np.isfinite(values).all():
        raise ValueError("response values must be finite")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("response epsilon must be finite and non-negative")
    magnitude = np.abs(values)
    magnitude[magnitude <= epsilon] = 0.0
    column_activity = magnitude @ mapping.target_to_column
    total = float(np.sum(column_activity))
    if total <= 0.0:
        return np.zeros(len(mapping.columns), dtype=np.float64)
    return np.asarray(column_activity / total, dtype=np.float64)


def activity_centroid(
    column_activity: np.ndarray,
    coordinates: np.ndarray,
) -> float | None:
    activity = np.asarray(column_activity, dtype=np.float64)
    position = np.asarray(coordinates, dtype=np.float64)
    if activity.ndim != 1 or activity.shape != position.shape:
        raise ValueError("centroid activity and coordinates must align")
    total = float(np.sum(activity))
    if total <= 0.0:
        return None
    return float(np.dot(activity, position) / total)


def pooled_profile(
    column_activity: np.ndarray,
    coordinates: np.ndarray,
    *,
    bin_count: int,
) -> FloatArray:
    """Pool normalized backprojected activity into fixed screen-coordinate bins."""

    if bin_count < 2:
        raise ValueError("profile_bin_count must be at least two")
    activity = np.asarray(column_activity, dtype=np.float64)
    position = np.asarray(coordinates, dtype=np.float64)
    if activity.ndim != 1 or activity.shape != position.shape:
        raise ValueError("profile activity and coordinates must align")
    if np.any(position < 0.0) or np.any(position > 1.0):
        raise ValueError("profile coordinates must stay inside [0, 1]")
    indices = np.minimum(
        (position * bin_count).astype(np.int64),
        bin_count - 1,
    )
    profile = np.zeros(bin_count, dtype=np.float64)
    np.add.at(profile, indices, activity)
    total = float(np.sum(profile))
    if total > 0.0:
        profile /= total
    return profile


def fit_affine(feature: np.ndarray, target: np.ndarray) -> AffineModel:
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("affine fit requires aligned one-dimensional samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("affine fit inputs must be finite")
    design = np.column_stack((x, np.ones_like(x)))
    coefficient, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return AffineModel(
        slope=float(coefficient[0]),
        intercept=float(coefficient[1]),
    )


def predict_affine(model: AffineModel, feature: np.ndarray) -> FloatArray:
    values = np.asarray(feature, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("affine prediction features must be finite and 1D")
    return np.asarray(
        model.slope * values + model.intercept,
        dtype=np.float64,
    )


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    base_config_path = ROOT / str(config["base_config"])
    base = load_base_config(base_config_path)
    train_positions = tuple(float(value) for value in config["train_positions"])
    test_positions = tuple(float(value) for value in config["test_positions"])
    _validate_positions(train_positions, test_positions)

    graph = MemoryMappedConnectome.load(data_directory)
    retinotopy_config_path = ROOT / str(base["retinotopy_config"])
    workbook_path = ROOT / str(base["optic_workbook"])
    directions_path = ROOT / str(base["measured_directions"])
    eye_map_manifest_path = ROOT / str(base["eye_map_manifest"])
    eye_map_lock = _verify_measured_directions(
        directions_path,
        eye_map_manifest_path,
    )
    columns = load_optic_columns(workbook_path)
    assignments = infer_r1r6_columns(
        graph,
        columns,
        load_retinotopy_config(retinotopy_config_path),
    )
    directions = load_measured_column_directions(directions_path)
    screen_spec = base["screen"]
    screen = VirtualScreenConfig(
        horizontal_fov_deg=float(screen_spec["horizontal_fov_deg"]),
        vertical_fov_deg=float(screen_spec["vertical_fov_deg"]),
    )
    projected_columns = {
        projection.column: (float(projection.u), float(projection.v))
        for projection in project_directions_to_screen(directions, screen)
        if projection.visible
        and projection.u is not None
        and projection.v is not None
    }
    source_groups = _source_groups(
        assignments,
        base,
        set(projected_columns),
    )
    cartridges = resolve_cartridge_sources(graph, source_groups, columns)
    _validate_lamina_signs(graph, cartridges)
    source_columns = tuple(cartridge.column for cartridge in cartridges)
    clamped = tuple(
        sorted(
            {
                index
                for cartridge in cartridges
                for index in (
                    *cartridge.r1r6_indices,
                    cartridge.l1_index,
                    cartridge.l2_index,
                    cartridge.l3_index,
                )
                if index is not None
            }
        )
    )
    projections = _lamina_projections(graph, cartridges)
    downstream_watch = _top_downstream_targets(
        projections,
        set(clamped),
        limit=int(config["feature"]["downstream_target_count"]),
    )
    backprojection = build_anatomical_backprojection(
        projections,
        downstream_watch,
        {
            column: projected_columns[column]
            for column in source_columns
        },
    )

    timing = base["timing"]
    lif_config = ShiuLIFConfig(dt_ms=float(timing["neural_dt_ms"]))
    runtime = base["runtime"]
    simulator = SparseLIFSimulator(
        graph,
        clamped_indices=clamped,
        config=lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
    )
    simulator.prepare()
    graded_config = _graded_config(base, lif_config)
    duration_s = (
        int(timing["baseline_frames"])
        + int(timing["stimulus_frames"])
        + int(timing["recovery_frames"])
    ) / float(timing["frame_rate_hz"])

    axes: dict[str, Any] = {}
    started = time.perf_counter()
    for axis_value in config["axes"]:
        axis = cast(Axis, str(axis_value))
        positions = train_positions + test_positions
        samples = [
            _run_position(
                axis,
                position,
                graph,
                simulator,
                base,
                directions,
                screen,
                source_columns,
                source_groups,
                cartridges,
                projections,
                downstream_watch,
                graded_config,
                lif_config,
                duration_s,
            )
            for position in positions
        ]
        axes[axis] = _decode_axis(
            axis,
            samples,
            backprojection,
            train_count=len(train_positions),
            train_positions=train_positions,
            test_positions=test_positions,
            config=config,
        )

    gates = _evaluate_gates(axes, config)
    target_centroid_u = backprojection.target_to_column @ backprojection.u
    target_centroid_v = backprojection.target_to_column @ backprojection.v
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Downstream subthreshold state is backprojected through fixed absolute "
            "L1/L2/L3 anatomical contacts onto measured visual columns. Position is "
            "then read from a one-dimensional activity centroid or a four-bin pooled "
            "profile. No screen coordinates are attached to individual neural "
            "responses during fitting."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "base_config_sha256": file_sha256(base_config_path),
        "eye_map_lock": eye_map_lock,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "frontend": {
            "visible_column_count": len(cartridges),
            "clamped_neuron_count": len(clamped),
            "downstream_watch_count": len(downstream_watch),
        },
        "topography": {
            "column_count": len(backprojection.columns),
            "minimum_target_contact_weight": float(
                np.min(backprojection.target_contact_weight)
            ),
            "median_target_contact_weight": float(
                np.median(backprojection.target_contact_weight)
            ),
            "target_rf_u_range": [
                float(np.min(target_centroid_u)),
                float(np.max(target_centroid_u)),
            ],
            "target_rf_v_range": [
                float(np.min(target_centroid_v)),
                float(np.max(target_centroid_v)),
            ],
        },
        "axes": axes,
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
    }


def _decode_axis(
    axis: Axis,
    samples: list[dict[str, Any]],
    mapping: AnatomicalBackprojection,
    *,
    train_count: int,
    train_positions: tuple[float, ...],
    test_positions: tuple[float, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["feature"]["response_epsilon"])
    bin_count = int(config["feature"]["profile_bin_count"])
    coordinates = mapping.coordinates(axis)
    y_train = np.asarray(train_positions, dtype=np.float64)
    y_test = np.asarray(test_positions, dtype=np.float64)
    ridge_config = config["ridge"]
    feature_results: dict[str, Any] = {}

    for feature_name in ("voltage", "drive", "spike"):
        column_activity = [
            backproject_response(
                sample[feature_name],
                mapping,
                epsilon=epsilon,
            )
            for sample in samples
        ]
        centroid_values: list[float] = []
        all_centroids_available = True
        for activity in column_activity:
            value = activity_centroid(activity, coordinates)
            if value is None:
                all_centroids_available = False
                centroid_values.append(math.nan)
            else:
                centroid_values.append(value)
        centroids = np.asarray(centroid_values, dtype=np.float64)
        profiles = np.vstack(
            [
                pooled_profile(
                    activity,
                    coordinates,
                    bin_count=bin_count,
                )
                for activity in column_activity
            ]
        )

        centroid_result: dict[str, Any]
        if all_centroids_available:
            centroid_model = fit_affine(
                centroids[:train_count],
                y_train,
            )
            centroid_train = predict_affine(
                centroid_model,
                centroids[:train_count],
            )
            centroid_test = predict_affine(
                centroid_model,
                centroids[train_count:],
            )
            centroid_result = {
                "model": {
                    "slope": centroid_model.slope,
                    "intercept": centroid_model.intercept,
                },
                "raw_centroids": [float(value) for value in centroids],
                "train": _prediction_metrics(y_train, centroid_train),
                "test": _prediction_metrics(y_test, centroid_test),
                "test_pairs": [
                    {
                        "actual": float(actual),
                        "centroid": float(centroid),
                        "predicted": float(predicted),
                    }
                    for actual, centroid, predicted in zip(
                        y_test,
                        centroids[train_count:],
                        centroid_test,
                        strict=True,
                    )
                ],
            }
        else:
            centroid_result = {
                "model": None,
                "raw_centroids": [
                    None if not math.isfinite(value) else float(value)
                    for value in centroids
                ],
                "train": None,
                "test": None,
                "test_pairs": [],
            }

        profile_model = fit_ridge(
            profiles[:train_count],
            y_train,
            alpha=float(ridge_config["alpha"]),
            epsilon=float(ridge_config["standardization_epsilon"]),
        )
        profile_train = predict_ridge(
            profile_model,
            profiles[:train_count],
        )
        profile_test = predict_ridge(
            profile_model,
            profiles[train_count:],
        )
        feature_results[feature_name] = {
            "centroid_affine": centroid_result,
            "pooled_profile": {
                "bin_count": bin_count,
                "active_feature_count": profile_model.active_feature_count,
                "profiles": [
                    [float(value) for value in row]
                    for row in profiles
                ],
                "train": _prediction_metrics(y_train, profile_train),
                "test": _prediction_metrics(y_test, profile_test),
                "test_pairs": [
                    {
                        "actual": float(actual),
                        "predicted": float(predicted),
                    }
                    for actual, predicted in zip(
                        y_test,
                        profile_test,
                        strict=True,
                    )
                ],
            },
        }

    return {
        "axis": axis,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "network_spike_counts": [
            int(sample["total_network_spikes"]) for sample in samples
        ],
        "features": feature_results,
    }


def _evaluate_gates(
    axes: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gates"]
    checks: dict[str, bool] = {}
    for axis, result in axes.items():
        metrics = result["features"]["voltage"]["centroid_affine"]["test"]
        if metrics is None:
            checks[f"{axis}_voltage_centroid_available"] = False
            continue
        correlation = metrics["correlation"]
        checks[f"{axis}_voltage_centroid_mae"] = (
            float(metrics["mae"])
            <= float(specification["maximum_voltage_centroid_test_mae"])
        )
        checks[f"{axis}_voltage_centroid_correlation"] = (
            correlation is not None
            and float(correlation)
            >= float(
                specification["minimum_voltage_centroid_test_correlation"]
            )
        )
        if bool(
            specification[
                "require_voltage_centroid_predictions_monotonic"
            ]
        ):
            checks[f"{axis}_voltage_centroid_monotonic"] = bool(
                metrics["monotonic"]
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": specification,
    }


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
                "gates_passed": result["gates"]["passed"],
                "horizontal_voltage_centroid_test": result["axes"][
                    "horizontal"
                ]["features"]["voltage"]["centroid_affine"]["test"],
                "vertical_voltage_centroid_test": result["axes"][
                    "vertical"
                ]["features"]["voltage"]["centroid_affine"]["test"],
                "horizontal_voltage_profile_test": result["axes"][
                    "horizontal"
                ]["features"]["voltage"]["pooled_profile"]["test"],
                "vertical_voltage_profile_test": result["axes"][
                    "vertical"
                ]["features"]["voltage"]["pooled_profile"]["test"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
