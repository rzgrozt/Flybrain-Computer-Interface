"""Sensitivity sweep for anatomically restricted graded visual relays."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_pathway import run_localization

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-graded-relay-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported graded-relay sensitivity schema")
    return payload


def _summary(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    motor_nonzero_drive = False
    motor_spikes = False
    max_motor_drive_mv = 0.0
    for response in result["responses"]:
        distance = int(response["distance"])
        for axis in response["axes"].values():
            layers = axis["layers"]
            if len(layers) < distance:
                continue
            motor = layers[distance - 1]
            motor_nonzero_drive = motor_nonzero_drive or (
                int(motor["nonzero_drive_neuron_count"]) > 0
            )
            motor_spikes = motor_spikes or int(motor["total_spikes"]) > 0
            max_motor_drive_mv = max(
                max_motor_drive_mv,
                float(motor["max_abs_drive_mv"]),
            )
    max_network_spikes = max(
        count
        for counts in result["network_spike_counts"].values()
        for count in counts
    )
    gates = config["gates"]
    checks = {
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
        "motor_nonzero_drive": motor_nonzero_drive,
        "motor_spikes": motor_spikes,
        "max_motor_drive_mv": max_motor_drive_mv,
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
    superclasses = tuple(str(value) for value in config["relay_superclasses"])
    gains = tuple(float(value) for value in config["graded_relay_gain_values"])
    activation_scale = float(config["graded_relay_activation_scale_mv"])
    if not superclasses:
        raise ValueError("relay_superclasses cannot be empty")
    if any(value <= 0.0 for value in gains):
        raise ValueError("graded relay gains must be positive")
    if activation_scale <= 0.0:
        raise ValueError("graded relay activation scale must be positive")

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    baseline = run_localization(data_directory, localization_config)
    runs.append(baseline)
    summaries.append(_summary(baseline, config))

    for gain in gains:
        result = run_localization(
            data_directory,
            localization_config,
            graded_relay_superclasses=superclasses,
            graded_relay_gain=gain,
            graded_relay_activation_scale_mv=activation_scale,
        )
        runs.append(result)
        summaries.append(_summary(result, config))

    passing = [
        item["intervention"]
        for item in summaries
        if bool(item["passed"])
    ]
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Selected shortest-path optic-lobe and visual relay neurons retain the "
            "canonical LIF state but additionally emit a depolarization-gated graded "
            "release along their real MaleCNS outgoing chemical edges. Motor neurons "
            "are never graded sources. This is a sensitivity analysis rather than a "
            "replacement canonical model."
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
