"""Isolate float64 subnormal costs in the fused LIF state kernel."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.kernels import (
    advance_state_numba,
    advance_state_numba_zero_subnormal,
)


def diagnose_subnormal_cost(
    *, neuron_count: int = 166_700, steps: int = 100, repeats: int = 7
) -> dict[str, Any]:
    """Compare matched kernel timings with no graph or event propagation."""

    if neuron_count <= 0 or steps <= 0 or repeats <= 0:
        raise ValueError("neuron_count, steps, and repeats must be positive")
    config = ShiuLIFConfig()
    membrane_decay = np.exp(-config.dt_ms / config.membrane_tau_ms)
    synapse_decay = np.exp(-config.dt_ms / config.synapse_tau_ms)
    drive_coupling = (
        config.synapse_tau_ms
        / (config.synapse_tau_ms - config.membrane_tau_ms)
        * (synapse_decay - membrane_decay)
    )
    smallest_normal = np.finfo(np.float64).tiny
    zero_policy_cutoff = smallest_normal / synapse_decay
    cases = (
        ("zero", 0.0),
        ("ordinary-normal", 1e-100),
        ("smallest-normal", smallest_normal),
        ("subnormal", np.nextafter(0.0, 1.0) * 1024),
    )

    voltage = np.full(neuron_count, config.resting_mv, dtype=np.float64)
    refractory = np.zeros(neuron_count, dtype=np.int64)
    spike_buffer = np.empty(neuron_count, dtype=np.int64)
    refractory_indices = np.empty(neuron_count, dtype=np.int64)
    refractory_drive = np.empty(neuron_count, dtype=np.float64)
    advance_state_numba(
        voltage,
        np.zeros(neuron_count, dtype=np.float64),
        refractory,
        spike_buffer,
        refractory_indices,
        refractory_drive,
        config.resting_mv,
        config.threshold_mv,
        config.tonic_bias_mv,
        membrane_decay,
        synapse_decay,
        drive_coupling,
    )

    results: list[dict[str, Any]] = []
    for policy, cutoff in (("preserve", 0.0), ("zero", zero_policy_cutoff)):
        for name, initial_drive in cases:
            timings: list[float] = []
            ending_drive = initial_drive
            ending_voltage = config.resting_mv
            for _ in range(repeats):
                voltage.fill(config.resting_mv)
                drive = np.full(neuron_count, initial_drive, dtype=np.float64)
                started = perf_counter()
                for _ in range(steps):
                    arguments = (
                        voltage,
                        drive,
                        refractory,
                        spike_buffer,
                        refractory_indices,
                        refractory_drive,
                        config.resting_mv,
                        config.threshold_mv,
                        config.tonic_bias_mv,
                        membrane_decay,
                        synapse_decay,
                        drive_coupling,
                    )
                    if policy == "zero":
                        advance_state_numba_zero_subnormal(*arguments, cutoff)
                    else:
                        advance_state_numba(*arguments)
                timings.append(perf_counter() - started)
                ending_drive = float(drive[0])
                ending_voltage = float(voltage[0])
            results.append(
                {
                    "policy": policy,
                    "value_class": name,
                    "initial_drive_mv": initial_drive,
                    "ending_drive_mv": ending_drive,
                    "ending_voltage_mv": ending_voltage,
                    "seconds": _summary(timings),
                }
            )

    preserve_zero = _median_for(results, "preserve", "zero")
    preserve_subnormal = _median_for(results, "preserve", "subnormal")
    zero_subnormal = _median_for(results, "zero", "subnormal")
    return {
        "diagnostic_schema_version": 1,
        "scope": "fused state kernel only; no graph, propagation, or recording",
        "neuron_count": neuron_count,
        "steps": steps,
        "repeats": repeats,
        "lif_config": asdict(config),
        "smallest_normal_float64": smallest_normal,
        "zero_policy_cutoff_mv": zero_policy_cutoff,
        "results": results,
        "ratios": {
            "preserved_subnormal_vs_zero": preserve_subnormal / preserve_zero,
            "zero_policy_vs_preserved_subnormal": zero_subnormal / preserve_subnormal,
        },
    }


def _median_for(results: list[dict[str, Any]], policy: str, value_class: str) -> float:
    match = next(
        result
        for result in results
        if result["policy"] == policy and result["value_class"] == value_class
    )
    return float(match["seconds"]["median_seconds"])


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "repeats": len(values),
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=166_700)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = diagnose_subnormal_cost(
        neuron_count=arguments.neurons,
        steps=arguments.steps,
        repeats=arguments.repeats,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
