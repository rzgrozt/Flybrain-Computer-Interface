"""Decode unseen screen position from canonical subthreshold MaleCNS state."""

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
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.experiments.retinotopy import (
    infer_r1r6_columns,
)
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    _lamina_projections,
    _make_lamina_drive,
    _top_downstream_targets,
    _validate_lamina_signs,
    resolve_cartridge_sources,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    load_config as load_base_config,
)
from flybrain_interface.experiments.spatial_neural_vision import (
    _condition_frames,
    _source_column_intensities,
    _source_groups,
    _total_frames,
    _verify_measured_directions,
)
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionConfig,
    VirtualScreenConfig,
    encode_screen_sequence,
    load_measured_column_directions,
    project_directions_to_screen,
    simulate_graded_early_vision_channels,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import ChunkRecording

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-subthreshold-readout-v1.json"
Axis = Literal["horizontal", "vertical"]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RidgeModel:
    """Tiny standardized dual-form ridge regressor."""

    feature_mean: FloatArray
    feature_scale: FloatArray
    active_mask: BoolArray
    training_standardized: FloatArray
    dual_weights: FloatArray
    target_mean: float
    alpha: float

    @property
    def active_feature_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial-subthreshold config schema")
    return payload


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> RidgeModel:
    """Fit a tiny ridge model without external ML dependencies."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("ridge features and targets must have aligned sample axes")
    if x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("ridge requires at least two samples and one feature")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ridge alpha must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("standardization epsilon must be finite and positive")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    active = scale > epsilon
    standardized: FloatArray = np.empty(
        (x.shape[0], int(np.count_nonzero(active))),
        dtype=np.float64,
    )
    if np.any(active):
        standardized[:] = (x[:, active] - mean[active]) / scale[active]
    target_mean = float(np.mean(y))
    centered = y - target_mean
    dual: FloatArray
    if standardized.shape[1] == 0:
        dual = np.zeros(x.shape[0], dtype=np.float64)
    else:
        gram = standardized @ standardized.T
        dual = np.asarray(
            np.linalg.solve(
                gram + alpha * np.eye(x.shape[0], dtype=np.float64),
                centered,
            ),
            dtype=np.float64,
        )

    return RidgeModel(
        feature_mean=mean,
        feature_scale=scale,
        active_mask=active,
        training_standardized=standardized,
        dual_weights=dual,
        target_mean=target_mean,
        alpha=alpha,
    )


def predict_ridge(model: RidgeModel, features: np.ndarray) -> FloatArray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.feature_mean.size:
        raise ValueError("prediction features do not match ridge feature dimension")
    if not np.isfinite(x).all():
        raise ValueError("prediction features must be finite")
    if model.active_feature_count == 0:
        return np.full(x.shape[0], model.target_mean, dtype=np.float64)
    standardized = (
        x[:, model.active_mask] - model.feature_mean[model.active_mask]
    ) / model.feature_scale[model.active_mask]
    prediction = (
        model.target_mean
        + standardized
        @ model.training_standardized.T
        @ model.dual_weights
    )
    return np.asarray(prediction, dtype=np.float64)


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
    visible_columns = {
        projection.column
        for projection in project_directions_to_screen(directions, screen)
        if projection.visible
    }
    source_groups = _source_groups(assignments, base, visible_columns)
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
    duration_s = _total_frames(base) / float(timing["frame_rate_hz"])

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
            train_count=len(train_positions),
            train_positions=train_positions,
            test_positions=test_positions,
            config=config,
        )

    gates = _evaluate_gates(axes, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Held-out screen position is decoded from downstream subthreshold state "
            "using a tiny ridge readout while the canonical graded-lamina frontend, "
            "50 Hz equivalent gain and 256-neuron anatomical watchlist remain fixed."
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
        "axes": axes,
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
    }


def _run_position(
    axis: Axis,
    position: float,
    graph: MemoryMappedConnectome,
    simulator: SparseLIFSimulator,
    base: dict[str, Any],
    directions: tuple[Any, ...],
    screen: VirtualScreenConfig,
    source_columns: tuple[str, ...],
    source_groups: dict[str, tuple[int, ...]],
    cartridges: tuple[Any, ...],
    projections: dict[Any, Any],
    downstream_watch: tuple[int, ...],
    graded_config: GradedEarlyVisionConfig,
    lif_config: ShiuLIFConfig,
    duration_s: float,
    temporal_bins: int = 0,
) -> dict[str, Any]:
    condition = {
        "name": f"dark_{axis}_{position:.4f}",
        "kind": "dark",
        "axis": axis,
        "position": position,
    }
    frames = _condition_frames(condition, base)
    encoding = encode_screen_sequence(
        frames,
        directions,
        screen,
        frame_rate_hz=float(base["timing"]["frame_rate_hz"]),
    )
    intensities = _source_column_intensities(
        encoding,
        source_columns,
        background=float(base["screen"]["background_intensity"]),
    )
    graded = simulate_graded_early_vision_channels(intensities, graded_config)
    projected = _make_lamina_drive(
        graded,
        cartridges,
        projections,
        base,
        lif_config,
    )
    if projected is None:
        raise RuntimeError("subthreshold readout stimulus created no projected drive")

    simulator.reset()
    result = simulator.advance_chunk(
        duration_s=duration_s,
        recording=ChunkRecording(
            watched_indices=downstream_watch,
            include_neuron_counts=True,
            max_spike_events=0,
        ),
        projected_drive=projected,
    )
    assert result.neuron_spike_counts is not None
    window = _feature_window(result.sample_times_s, base)
    voltage = _signed_peak(
        result.voltage_mv[window] - simulator.config.resting_mv
    )
    drive = _signed_peak(result.synaptic_drive_mv[window])
    spike = result.neuron_spike_counts[
        np.asarray(downstream_watch, dtype=np.int64)
    ].astype(np.float64)
    payload: dict[str, Any] = {
        "position": position,
        "voltage": voltage,
        "drive": drive,
        "spike": spike,
        "total_network_spikes": result.total_spikes,
        "frame_sha256": encoding.frame_sha256,
    }
    if temporal_bins:
        if temporal_bins <= 0:
            raise ValueError("temporal_bins must be positive when enabled")
        voltage_window = result.voltage_mv[window] - simulator.config.resting_mv
        drive_window = result.synaptic_drive_mv[window]
        if voltage_window.shape[0] < temporal_bins:
            raise ValueError("temporal bin count exceeds available feature samples")
        index_bins = np.array_split(
            np.arange(voltage_window.shape[0], dtype=np.int64),
            temporal_bins,
        )
        payload["temporal_voltage"] = np.vstack(
            [_signed_peak(voltage_window[indices]) for indices in index_bins]
        )
        payload["temporal_drive"] = np.vstack(
            [_signed_peak(drive_window[indices]) for indices in index_bins]
        )
    return payload


def _decode_axis(
    axis: Axis,
    samples: list[dict[str, Any]],
    *,
    train_count: int,
    train_positions: tuple[float, ...],
    test_positions: tuple[float, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    ridge = config["ridge"]
    alpha = float(ridge["alpha"])
    epsilon = float(ridge["standardization_epsilon"])
    y_train = np.asarray(train_positions, dtype=np.float64)
    y_test = np.asarray(test_positions, dtype=np.float64)
    feature_results: dict[str, Any] = {}

    for feature_name in ("voltage", "drive", "spike"):
        x = np.vstack([sample[feature_name] for sample in samples])
        x_train = x[:train_count]
        x_test = x[train_count:]
        model = fit_ridge(
            x_train,
            y_train,
            alpha=alpha,
            epsilon=epsilon,
        )
        train_prediction = predict_ridge(model, x_train)
        test_prediction = predict_ridge(model, x_test)
        feature_results[feature_name] = {
            "active_feature_count": model.active_feature_count,
            "train": _prediction_metrics(y_train, train_prediction),
            "test": _prediction_metrics(y_test, test_prediction),
            "train_pairs": [
                {"actual": float(actual), "predicted": float(predicted)}
                for actual, predicted in zip(
                    y_train,
                    train_prediction,
                    strict=True,
                )
            ],
            "test_pairs": [
                {"actual": float(actual), "predicted": float(predicted)}
                for actual, predicted in zip(
                    y_test,
                    test_prediction,
                    strict=True,
                )
            ],
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


def _graded_config(
    base: dict[str, Any],
    lif_config: ShiuLIFConfig,
) -> GradedEarlyVisionConfig:
    timing = base["timing"]
    model = base["graded_model"]
    return GradedEarlyVisionConfig(
        frame_rate_hz=float(timing["frame_rate_hz"]),
        neural_dt_ms=lif_config.dt_ms,
        background_luminance=float(model["background_luminance"]),
        tonic_release_fraction=float(model["tonic_release_fraction"]),
        photoreceptor_tau_ms=float(model["photoreceptor_tau_ms"]),
        transient_fast_tau_ms=float(model["transient_fast_tau_ms"]),
        transient_slow_tau_ms=float(model["transient_slow_tau_ms"]),
        l3_tau_ms=float(model["l3_tau_ms"]),
        l1_gain_mv=float(model["l1_gain_mv"]),
        l2_gain_mv=float(model["l2_gain_mv"]),
        l2_slow_weight=float(model["l2_slow_weight"]),
        l3_gain_mv=float(model["l3_gain_mv"]),
    )


def _feature_window(
    sample_times_s: np.ndarray,
    base: dict[str, Any],
) -> np.ndarray:
    timing = base["timing"]
    start = int(timing["baseline_frames"]) / float(timing["frame_rate_hz"])
    return sample_times_s >= start


def _signed_peak(trace: np.ndarray) -> np.ndarray:
    if trace.ndim != 2 or trace.shape[0] == 0:
        raise ValueError("signed-peak trace must have shape (time, features)")
    positive = np.max(trace, axis=0)
    negative = np.min(trace, axis=0)
    return np.where(np.abs(positive) >= np.abs(negative), positive, negative)


def _prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    error = predicted - actual
    correlation: float | None
    if (
        actual.size >= 2
        and float(np.std(actual)) > 0.0
        and float(np.std(predicted)) > 1e-12
    ):
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        correlation = None
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "correlation": correlation,
        "monotonic": bool(
            all(
                after > before
                for before, after in zip(predicted, predicted[1:])
            )
        ),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
    }


def _evaluate_gates(
    axes: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gates"]
    checks: dict[str, bool] = {}
    for axis, result in axes.items():
        metrics = result["features"]["voltage"]["test"]
        correlation = metrics["correlation"]
        checks[f"{axis}_voltage_mae"] = (
            float(metrics["mae"])
            <= float(specification["maximum_voltage_test_mae"])
        )
        checks[f"{axis}_voltage_correlation"] = (
            correlation is not None
            and float(correlation)
            >= float(specification["minimum_voltage_test_correlation"])
        )
        if bool(specification["require_voltage_predictions_monotonic"]):
            checks[f"{axis}_voltage_monotonic"] = bool(metrics["monotonic"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": specification,
    }


def _validate_positions(
    train: tuple[float, ...],
    test: tuple[float, ...],
) -> None:
    if len(train) < 3 or len(test) < 2:
        raise ValueError("readout requires at least three train and two test positions")
    if set(train) & set(test):
        raise ValueError("train and test positions must be disjoint")
    if any(not 0.0 < value < 1.0 for value in (*train, *test)):
        raise ValueError("all readout positions must be inside (0, 1)")
    if tuple(sorted(train)) != train or tuple(sorted(test)) != test:
        raise ValueError("readout positions must be strictly sorted")


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
                "horizontal_voltage_test": result["axes"]["horizontal"][
                    "features"
                ]["voltage"]["test"],
                "vertical_voltage_test": result["axes"]["vertical"][
                    "features"
                ]["voltage"]["test"],
                "horizontal_spike_test": result["axes"]["horizontal"][
                    "features"
                ]["spike"]["test"],
                "vertical_spike_test": result["axes"]["vertical"][
                    "features"
                ]["spike"]["test"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
