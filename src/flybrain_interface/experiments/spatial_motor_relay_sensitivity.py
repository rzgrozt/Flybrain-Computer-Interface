"""Targeted hop-1 excitability sensitivity for visual-to-motor propagation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_pathway import run_localization

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-hop1-excitability-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported hop1 excitability sensitivity schema")
    return payload


def _summarize_run(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hop1_total_spikes = 0
    hop2_nonzero_drive = False
    motor_nonzero_drive = False
    motor_spikes = False
    for response in result["responses"]:
        distance = int(response["distance"])
        for axis in response["axes"].values():
            layers = axis["layers"]
            if layers:
                hop1_total_spikes = max(
                    hop1_total_spikes,
                    int(layers[0]["total_spikes"]),
                )
            if len(layers) >= 2:
                hop2_nonzero_drive = hop2_nonzero_drive or (
                    int(layers[1]["nonzero_drive_neuron_count"]) > 0
                )
            if len(layers) >= distance:
                motor = layers[distance - 1]
                motor_nonzero_drive = motor_nonzero_drive or (
                    int(motor["nonzero_drive_neuron_count"]) > 0
                )
                motor_spikes = motor_spikes or int(motor["total_spikes"]) > 0

    max_network_spikes = max(
        count
        for counts in result["network_spike_counts"].values()
        for count in counts
    )
    gates = config["gates"]
    checks = {
        "hop2_nonzero_drive": (
            hop2_nonzero_drive
            if bool(gates["require_hop2_nonzero_drive"])
            else True
        ),
        "motor_nonzero_drive": (
            motor_nonzero_drive
            if bool(gates["require_motor_nonzero_drive"])
            else True
        ),
        "network_spike_budget": max_network_spikes
        <= int(gates["maximum_network_spikes_per_run"]),
    }
    return {
        "intervention": result["intervention"],
        "hop1_max_total_spikes": hop1_total_spikes,
        "hop2_nonzero_drive": hop2_nonzero_drive,
        "motor_nonzero_drive": motor_nonzero_drive,
        "motor_spikes": motor_spikes,
        "max_network_spikes_per_run": max_network_spikes,
        "checks": checks,
        "passed": all(checks.values()),
        "wall_seconds": float(result["total_wall_seconds"]),
    }


def run_sensitivity(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    localization_config = ROOT / str(config["localization_config"])

    tonic_values = tuple(float(value) for value in config["tonic_bias_values_mv"])
    threshold_values = tuple(float(value) for value in config["threshold_values_mv"])
    if any(value <= 0.0 for value in tonic_values):
        raise ValueError("targeted tonic biases must be positive")
    if any(value <= -52.0 or value >= -45.0 for value in threshold_values):
        raise ValueError(
            "targeted thresholds must lie strictly between rest and baseline threshold"
        )

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    baseline = run_localization(data_directory, localization_config)
    runs.append(baseline)
    summaries.append(_summarize_run(baseline, config))

    for bias in tonic_values:
        result = run_localization(
            data_directory,
            localization_config,
            hop1_tonic_bias_mv=bias,
        )
        runs.append(result)
        summaries.append(_summarize_run(result, config))

    for threshold in threshold_values:
        result = run_localization(
            data_directory,
            localization_config,
            hop1_threshold_mv=threshold,
        )
        runs.append(result)
        summaries.append(_summarize_run(result, config))

    passing = [
        summary["intervention"]
        for summary in summaries
        if bool(summary["passed"])
    ]
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Only neurons in the union of shortest effective hop-1 motor-path layers "
            "receive excitability changes. The canonical lamina drive, synaptic scale, "
            "connectome topology, motor populations and all non-hop1 neurons remain "
            "unchanged. This isolates whether the first all-or-none relay is the "
            "mechanistic propagation bottleneck."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "localization_config_sha256": file_sha256(localization_config),
        "summaries": summaries,
        "passing_interventions": passing,
        "gates": {
            "passed": bool(passing),
            "thresholds": config["gates"],
        },
        "runs": runs,
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
                "passing_interventions": result["passing_interventions"],
                "summaries": result["summaries"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
