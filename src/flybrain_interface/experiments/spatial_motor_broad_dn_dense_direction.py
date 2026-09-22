"""Dense binary direction readout with independent midpoint validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_binary_direction import (
    CLASS_NAMES,
    _fit_pipeline,
    _score_pipeline,
    _select_candidate,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _matrix,
    _samples_by_position,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-dn-dense-direction-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported dense direction schema")
    return payload


def _labels(positions: tuple[float, ...], center: float) -> np.ndarray:
    return np.asarray(
        [0 if position < center else 1 for position in positions],
        dtype=np.int64,
    )


def _evaluate(
    model: dict[str, Any],
    features: np.ndarray,
    positions: tuple[float, ...],
    center: float,
) -> dict[str, Any]:
    labels = _labels(positions, center)
    scores = _score_pipeline(model, features)
    predicted = np.argmax(scores, axis=1)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "pairs": [
            {
                "position": float(position),
                "expected": CLASS_NAMES[int(expected)],
                "predicted": CLASS_NAMES[int(prediction)],
                "scores": [float(value) for value in row_scores],
                "correct": bool(expected == prediction),
            }
            for position, expected, prediction, row_scores in zip(
                positions,
                labels,
                predicted,
                scores,
                strict=True,
            )
        ],
    }


def benchmark(
    training_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    validation_positions = tuple(
        float(v) for v in config["validation_positions"]
    )
    train_samples = _samples_by_position(training_artifact, axis)
    validation_samples = _samples_by_position(validation_artifact, axis)
    labels = _labels(train_positions, center)
    selection = _select_candidate(
        train_samples,
        train_positions,
        labels,
        config,
    )
    winner = selection["winner"]
    modality = str(winner["modality"])
    x_train = _matrix(train_samples, train_positions, modality)
    model = _fit_pipeline(
        x_train,
        labels,
        components=int(winner["components"]),
        alpha=float(winner["alpha"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    validation = _evaluate(
        model,
        _matrix(validation_samples, validation_positions, modality),
        validation_positions,
        center,
    )
    validation["passes"] = bool(
        validation["accuracy"]
        >= float(config["gates"]["minimum_validation_accuracy"])
    )
    return {
        "winner": winner,
        "selection": selection,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    training_artifact = json.loads(
        (ROOT / str(config["training_artifact"])).read_text(encoding="utf-8")
    )
    validation_artifact = json.loads(
        (ROOT / str(config["validation_artifact"])).read_text(encoding="utf-8")
    )
    result = benchmark(training_artifact, validation_artifact, config)
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "PCA/ridge hyperparameters are selected only from the dense directional "
            "training grid using leave-one-out accuracy. Independent midpoint neural "
            "states are used once for the validation gate. The readout receives only "
            "broad descending-neuron temporal state."
        ),
        "config": config,
        "benchmark": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "winner": result["winner"],
                "validation": result["validation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
