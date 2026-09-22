"""Benchmark linear versus tiny nonlinear readouts on descending-neuron state only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.experiments.spatial_motor_readout import run_characterization

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-dn-nonlinear-readout-v1.json"
POPULATION_KEYS = (
    "steer_left",
    "steer_right",
    "steer_left_secondary",
    "steer_right_secondary",
    "forward",
    "backward",
)
MEMBER_METRICS = (
    "signed_peak_voltage_delta_mv",
    "signed_peak_drive_mv",
    "spike_count",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DN nonlinear-readout schema")
    return payload


def _dn_features(run: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    names: list[str] = []
    for population_key in POPULATION_KEYS:
        members = run["populations"][population_key]["members"]
        for member in members:
            instance = str(member["instance"])
            body_id = int(member["body_id"])
            label = f"{population_key}:{instance}:{body_id}"
            for metric in MEMBER_METRICS:
                values.append(float(member[metric]))
                names.append(f"{label}:{metric}")
    return np.asarray(values, dtype=np.float64), tuple(names)


def _standardize_fit(
    features: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    active = scale > epsilon
    safe_scale = np.where(active, scale, np.inf)
    standardized = (features - mean) / safe_scale
    return standardized, mean, safe_scale


def _standardize_apply(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (features - mean) / scale
    return result


def _fit_linear_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    x_standard, mean, scale = _standardize_fit(x_train, epsilon=epsilon)
    design = np.column_stack((np.ones(x_standard.shape[0]), x_standard))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ y_train,
    )
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
    }


def _predict_linear(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x_standard = _standardize_apply(
        features,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    design = np.column_stack((np.ones(x_standard.shape[0]), x_standard))
    return design @ np.asarray(model["weights"], dtype=np.float64)


def _rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    squared = np.sum(delta * delta, axis=2)
    return np.exp(-gamma * squared)


def _fit_rbf_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> dict[str, Any]:
    if gamma <= 0.0:
        raise ValueError("RBF gamma must be positive")
    x_standard, mean, scale = _standardize_fit(x_train, epsilon=epsilon)
    kernel = _rbf_kernel(x_standard, x_standard, gamma=gamma)
    weights = np.linalg.solve(
        kernel + alpha * np.eye(kernel.shape[0], dtype=np.float64),
        y_train,
    )
    return {
        "mean": mean,
        "scale": scale,
        "centers": x_standard,
        "weights": weights,
        "gamma": gamma,
    }


def _predict_rbf(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x_standard = _standardize_apply(
        features,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    kernel = _rbf_kernel(
        x_standard,
        np.asarray(model["centers"], dtype=np.float64),
        gamma=float(model["gamma"]),
    )
    return kernel @ np.asarray(model["weights"], dtype=np.float64)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = predicted - actual
    side_correct = [
        bool(
            (expected < 0.5 and estimate < 0.5)
            or (expected > 0.5 and estimate > 0.5)
            or (expected == 0.5 and abs(estimate - 0.5) <= 1e-12)
        )
        for expected, estimate in zip(actual, predicted, strict=True)
    ]
    correlation = None
    if (
        actual.size >= 2
        and float(np.std(actual)) > 0.0
        and float(np.std(predicted)) > 1e-12
    ):
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "side_accuracy": float(sum(side_correct) / len(side_correct)),
        "correlation": correlation,
        "pairs": [
            {
                "actual": float(expected),
                "predicted": float(estimate),
                "side_correct": correct,
            }
            for expected, estimate, correct in zip(
                actual,
                predicted,
                side_correct,
                strict=True,
            )
        ],
    }


def _ordered_matrix(
    runs: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    by_position = {float(run["position"]): run for run in runs}
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for position in positions:
        if position not in by_position:
            raise ValueError(f"missing motor-state position: {position}")
        row, row_names = _dn_features(by_position[position])
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("DN feature ordering changed across runs")
        rows.append(row)
    assert names is not None
    return np.vstack(rows), names


def benchmark_from_characterization(
    characterization: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    axis = str(config["axis"])
    runs = characterization["axes"][axis]["runs"]
    train_positions = tuple(float(value) for value in config["train_positions"])
    test_positions = tuple(float(value) for value in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")

    x_train, feature_names = _ordered_matrix(runs, train_positions)
    x_test, test_feature_names = _ordered_matrix(runs, test_positions)
    if feature_names != test_feature_names:
        raise RuntimeError("train/test DN feature ordering mismatch")
    y_train = np.asarray(train_positions, dtype=np.float64)
    y_test = np.asarray(test_positions, dtype=np.float64)

    linear_spec = config["linear_ridge"]
    linear_model = _fit_linear_ridge(
        x_train,
        y_train,
        alpha=float(linear_spec["alpha"]),
        epsilon=float(linear_spec["standardization_epsilon"]),
    )
    linear_prediction = _predict_linear(linear_model, x_test)

    rbf_spec = config["rbf_ridge"]
    rbf_model = _fit_rbf_ridge(
        x_train,
        y_train,
        alpha=float(rbf_spec["alpha"]),
        gamma=float(rbf_spec["gamma"]),
        epsilon=float(rbf_spec["standardization_epsilon"]),
    )
    rbf_prediction = _predict_rbf(rbf_model, x_test)

    linear_metrics = _metrics(y_test, linear_prediction)
    rbf_metrics = _metrics(y_test, rbf_prediction)
    gates = config["gates"]
    for result in (linear_metrics, rbf_metrics):
        result["passes"] = bool(
            float(result["mae"]) <= float(gates["maximum_test_mae"])
            and float(result["side_accuracy"])
            >= float(gates["minimum_side_accuracy"])
        )

    return {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "linear_ridge": linear_metrics,
        "rbf_ridge": rbf_metrics,
    }


def run_benchmark(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    motor_config = ROOT / str(config["motor_config"])
    characterization = run_characterization(data_directory, motor_config)
    result = benchmark_from_characterization(characterization, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Both readouts receive only individual descending-neuron voltage, drive, "
            "and spike state from anatomically selected DNa02/DNg13/DNp09/MDN neurons. "
            "The RBF model is a tiny nonlinear calibration over DN-state similarity; "
            "neither model receives screen coordinates or topographic features."
        ),
        "config": config,
        "motor_characterization": characterization,
        "benchmark": result,
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
    args = parser.parse_args()

    result = run_benchmark(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "linear_ridge": result["benchmark"]["linear_ridge"],
                "rbf_ridge": result["benchmark"]["rbf_ridge"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
