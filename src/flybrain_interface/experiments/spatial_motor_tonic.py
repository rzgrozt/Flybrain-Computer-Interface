"""Predeclared tonic-excitability sweep for spatial visual-to-motor propagation."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_readout import (
    run_characterization,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-tonic-sensitivity-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial motor tonic-sensitivity schema")
    return payload


def _range(values: list[float]) -> float:
    return max(values) - min(values)


def _summarize_bias(
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    horizontal_runs = result["axes"]["horizontal"]["runs"]
    vertical_runs = result["axes"]["vertical"]["runs"]
    horizontal_values = [
        float(run["channels"]["steering_voltage_right_minus_left_mv"])
        for run in horizontal_runs
    ]
    vertical_values = [
        float(run["channels"]["vertical_voltage_backward_minus_forward_mv"])
        for run in vertical_runs
    ]
    horizontal_range = _range(horizontal_values)
    vertical_range = _range(vertical_values)
    max_spikes = max(
        int(run["total_network_spikes"])
        for axis in result["axes"].values()
        for run in axis["runs"]
    )
    gates = config["gates"]
    checks = {
        "horizontal_steering_range": horizontal_range
        >= float(gates["minimum_horizontal_steering_range_mv"]),
        "vertical_forward_backward_range": vertical_range
        >= float(gates["minimum_vertical_forward_backward_range_mv"]),
        "network_spike_budget": max_spikes
        <= int(gates["maximum_network_spikes_per_run"]),
    }
    return {
        "tonic_bias_mv": float(result["config"]["tonic_bias_mv"]),
        "horizontal_steering_range_mv": horizontal_range,
        "vertical_forward_backward_range_mv": vertical_range,
        "horizontal_steering_values_mv": horizontal_values,
        "vertical_forward_backward_values_mv": vertical_values,
        "max_network_spikes_per_run": max_spikes,
        "checks": checks,
        "passed": all(checks.values()),
        "wall_seconds": float(result["total_wall_seconds"]),
    }


def run_sensitivity(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    motor_config_path = ROOT / str(config["motor_config"])
    motor_config = json.loads(motor_config_path.read_text(encoding="utf-8"))
    biases = tuple(float(value) for value in config["tonic_bias_values_mv"])
    if not biases or tuple(sorted(set(biases))) != biases:
        raise ValueError("tonic bias values must be unique and sorted")
    if biases[0] != 0.0:
        raise ValueError("tonic sensitivity must include the zero-bias control first")

    started = time.perf_counter()
    detailed_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="flybrain-tonic-sweep-") as directory:
        temp_directory = Path(directory)
        for bias in biases:
            payload = copy.deepcopy(motor_config)
            payload["name"] = (
                f"{config['name']}-bias-{bias:.3f}".replace(".", "p")
            )
            payload["tonic_bias_mv"] = bias
            payload["compute_hops"] = False
            temporary_config = temp_directory / f"motor-{bias:.3f}.json"
            temporary_config.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = run_characterization(
                data_directory,
                temporary_config,
            )
            detailed_runs.append(result)
            summaries.append(_summarize_bias(result, config))

    passing_biases = [
        float(summary["tonic_bias_mv"])
        for summary in summaries
        if bool(summary["passed"])
    ]
    require_same = bool(config["gates"]["require_both_axes_same_bias"])
    passed = bool(passing_biases) if require_same else all(
        any(
            bool(summary["checks"][check])
            for summary in summaries
        )
        for check in (
            "horizontal_steering_range",
            "vertical_forward_backward_range",
        )
    )
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "This is a sensitivity analysis, not a replacement canonical model. "
            "The visual frontend, threshold, synaptic scale and topology remain "
            "unchanged while a uniform subthreshold tonic membrane bias is swept. "
            "Success requires visual-position-dependent continuous state in both "
            "anatomically grounded motor channels at the same bias without runaway."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "motor_config_sha256": file_sha256(motor_config_path),
        "summaries": summaries,
        "passing_biases_mv": passing_biases,
        "gates": {
            "passed": passed,
            "require_both_axes_same_bias": require_same,
            "thresholds": config["gates"],
        },
        "runs": detailed_runs,
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
    arguments = parser.parse_args()

    result = run_sensitivity(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "gates_passed": result["gates"]["passed"],
                "passing_biases_mv": result["passing_biases_mv"],
                "summaries": result["summaries"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
