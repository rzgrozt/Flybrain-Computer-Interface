"""Localize spatial-position decodability along visual-to-motor pathway hops."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_pathway import (
    PathwayLayers,
    _motor_targets,
    _pathway_records,
    _setup_visual_system,
    _watchlist,
)
from flybrain_interface.experiments.spatial_subthreshold_readout import (
    _prediction_metrics,
    _run_position,
    fit_ridge,
    predict_ridge,
)
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-hop-geometry-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial motor hop-geometry schema")
    return payload


def _side_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    center: float = 0.5,
) -> float:
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("side-accuracy inputs must be matching vectors")
    if actual.size == 0:
        raise ValueError("side-accuracy inputs cannot be empty")
    correct = 0
    for expected, estimate in zip(actual, predicted, strict=True):
        if expected < center and estimate < center:
            correct += 1
        elif expected > center and estimate > center:
            correct += 1
        elif expected == center and abs(estimate - center) <= 1e-12:
            correct += 1
    return correct / actual.size


def _decode_feature_block(
    features: np.ndarray,
    train_positions: tuple[float, ...],
    test_positions: tuple[float, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("feature block must be two-dimensional")
    train_count = len(train_positions)
    expected_rows = train_count + len(test_positions)
    if features.shape[0] != expected_rows:
        raise ValueError("feature block row count does not match positions")

    ridge = config["ridge"]
    model = fit_ridge(
        features[:train_count],
        np.asarray(train_positions, dtype=np.float64),
        alpha=float(ridge["alpha"]),
        epsilon=float(ridge["standardization_epsilon"]),
    )
    train_prediction = predict_ridge(model, features[:train_count])
    test_prediction = predict_ridge(model, features[train_count:])
    y_train = np.asarray(train_positions, dtype=np.float64)
    y_test = np.asarray(test_positions, dtype=np.float64)
    return {
        "active_feature_count": model.active_feature_count,
        "train": _prediction_metrics(y_train, train_prediction),
        "test": {
            **_prediction_metrics(y_test, test_prediction),
            "side_accuracy": _side_accuracy(y_test, test_prediction),
        },
        "test_pairs": [
            {"actual": float(actual), "predicted": float(predicted)}
            for actual, predicted in zip(y_test, test_prediction, strict=True)
        ],
    }


def _layer_decodability(
    layer: tuple[int, ...],
    samples: list[dict[str, Any]],
    watch_position: dict[int, int],
    train_positions: tuple[float, ...],
    test_positions: tuple[float, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    positions = np.asarray(
        [watch_position[index] for index in layer],
        dtype=np.int64,
    )
    feature_results: dict[str, Any] = {}
    for feature_name in ("voltage", "drive", "spike"):
        matrix = np.vstack(
            [
                np.asarray(sample[feature_name], dtype=np.float64)[positions]
                for sample in samples
            ]
        )
        feature_results[feature_name] = _decode_feature_block(
            matrix,
            train_positions,
            test_positions,
            config,
        )
    gates = config["gates"]
    max_mae = float(gates["maximum_test_mae"])
    min_side = float(gates["minimum_test_side_accuracy"])
    passing_features = [
        name
        for name, result in feature_results.items()
        if float(result["test"]["mae"]) <= max_mae
        and float(result["test"]["side_accuracy"]) >= min_side
    ]
    best_name = min(
        feature_results,
        key=lambda name: float(feature_results[name]["test"]["mae"]),
    )
    return {
        "neuron_count": len(layer),
        "features": feature_results,
        "best_feature": best_name,
        "best_test_mae": float(feature_results[best_name]["test"]["mae"]),
        "best_test_side_accuracy": float(
            feature_results[best_name]["test"]["side_accuracy"]
        ),
        "passing_features": passing_features,
        "passes_generalization": bool(passing_features),
    }


def _graded_relays(
    setup: Any,
    pathways: dict[int, PathwayLayers],
    superclasses: set[str],
) -> tuple[int, ...]:
    table = setup.graph.catalog.table
    return tuple(
        sorted(
            {
                neuron
                for pathway in pathways.values()
                for layer in pathway.layers[1:-1]
                for neuron in layer
                if table["superclass"][neuron].as_py() in superclasses
            }
        )
    )


def _first_failed_hop(layers: list[dict[str, Any]]) -> int | None:
    for layer in layers:
        if not bool(layer["passes_generalization"]):
            return int(layer["hop"])
    return None


def run_localization(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    pathway_config_path = ROOT / str(config["pathway_config"])
    pathway_config = json.loads(pathway_config_path.read_text(encoding="utf-8"))
    base_config_path = ROOT / str(pathway_config["base_config"])
    motor_config_path = ROOT / str(pathway_config["motor_config"])

    train_positions = tuple(float(value) for value in config["train_positions"])
    test_positions = tuple(float(value) for value in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")
    all_positions = (*train_positions, *test_positions)

    setup = _setup_visual_system(data_directory, base_config_path)
    motor_populations = _motor_targets(setup.graph, motor_config_path)
    pathway_records, pathways = _pathway_records(
        setup.graph,
        motor_populations,
        setup.lamina_sources,
        max_hops=int(pathway_config["maximum_path_hops"]),
        top_paths=int(pathway_config["top_paths_per_target"]),
    )
    watch_indices = _watchlist(pathways, setup.lamina_sources)
    if len(watch_indices) > int(pathway_config["maximum_watch_neurons"]):
        raise RuntimeError("hop-geometry watchlist exceeds configured bound")
    watch_position = {
        neuron_index: position
        for position, neuron_index in enumerate(watch_indices)
    }

    relay_spec = config["graded_relay"]
    graded_indices = _graded_relays(
        setup,
        pathways,
        {str(value) for value in relay_spec["superclasses"]},
    )
    runtime = setup.base["runtime"]
    simulator = SparseLIFSimulator(
        setup.graph,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
        graded_relay_indices=graded_indices,
        graded_relay_gain=float(relay_spec["gain"]),
        graded_relay_activation_scale_mv=float(
            relay_spec["activation_scale_mv"]
        ),
    )
    simulator.prepare()

    record_by_target = {
        int(record["target_neuron_index"]): record
        for record in pathway_records
        if bool(record["reachable"])
    }
    started = time.perf_counter()
    axes: dict[str, Any] = {}
    for axis in pathway_config["axes"]:
        samples = [
            _run_position(
                axis,
                position,
                setup.graph,
                simulator,
                setup.base,
                setup.directions,
                setup.screen,
                setup.source_columns,
                setup.source_groups,
                setup.cartridges,
                setup.projections,
                watch_indices,
                setup.graded_config,
                setup.lif_config,
                setup.duration_s,
            )
            for position in all_positions
        ]
        target_results: list[dict[str, Any]] = []
        for target_index, pathway in pathways.items():
            record = record_by_target[target_index]
            layers: list[dict[str, Any]] = []
            for hop, layer in enumerate(pathway.layers[1:], start=1):
                decoded = _layer_decodability(
                    layer,
                    samples,
                    watch_position,
                    train_positions,
                    test_positions,
                    config,
                )
                layers.append({"hop": hop, **decoded})
            target_results.append(
                {
                    "motor_population": record["motor_population"],
                    "target_neuron_index": target_index,
                    "target_body_id": record["target_body_id"],
                    "target_instance": record["target_instance"],
                    "distance": pathway.distance,
                    "layers": layers,
                    "first_failed_hop": _first_failed_hop(layers),
                }
            )
        axes[str(axis)] = {
            "samples": [
                {
                    "position": float(sample["position"]),
                    "total_network_spikes": int(sample["total_network_spikes"]),
                }
                for sample in samples
            ],
            "targets": target_results,
        }

    failure_histogram: dict[str, int] = {}
    for axis in axes.values():
        for target in axis["targets"]:
            key = (
                "none"
                if target["first_failed_hop"] is None
                else str(target["first_failed_hop"])
            )
            failure_histogram[key] = failure_histogram.get(key, 0) + 1

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Spatial position is decoded independently from voltage, drive and spike "
            "state at each shortest-path hop using disjoint train/test positions. "
            "The first hop that fails held-out position gates localizes where spatial "
            "geometry ceases to generalize toward the descending motor target."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "pathway_config_sha256": file_sha256(pathway_config_path),
        "graded_relay_neuron_count": len(graded_indices),
        "watch_count": len(watch_indices),
        "axes": axes,
        "first_failed_hop_histogram": failure_histogram,
        "total_wall_seconds": time.perf_counter() - started,
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

    result = run_localization(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "first_failed_hop_histogram": result[
                    "first_failed_hop_histogram"
                ],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
