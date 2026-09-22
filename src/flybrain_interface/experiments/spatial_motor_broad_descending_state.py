"""Record broad anatomically reachable descending-neuron state under spatial vision."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_pathway import (
    _setup_visual_system,
)
from flybrain_interface.experiments.spatial_motor_readout import (
    _effective_forward_distances,
    _shortest_effective_layers,
    resolve_motor_populations,
)
from flybrain_interface.experiments.spatial_subthreshold_readout import _run_position
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-descending-state-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad descending-state schema")
    return payload


def _graded_relay_indices(
    setup: Any,
    motor_config: dict[str, Any],
) -> tuple[int, ...]:
    spec = motor_config.get("graded_relay")
    if spec is None:
        return ()
    maximum_path_hops = int(spec.get("maximum_path_hops", 3))
    allowed_superclasses = {str(value) for value in spec["superclasses"]}
    excluded_types = {str(value) for value in spec.get("exclude_types", [])}
    populations = resolve_motor_populations(setup.graph.catalog.table, motor_config)
    forward = _effective_forward_distances(
        setup.graph,
        set(setup.lamina_sources),
        max_hops=maximum_path_hops,
    )
    relay_set: set[int] = set()
    for population in populations:
        for neuron_index in population.neuron_indices:
            layers = _shortest_effective_layers(
                setup.graph,
                forward,
                neuron_index,
                max_hops=maximum_path_hops,
            )
            if layers is None:
                continue
            for layer in layers[1:-1]:
                for relay_index in layer:
                    table = setup.graph.catalog.table
                    superclass = table["superclass"][relay_index].as_py()
                    type_name = table["type"][relay_index].as_py()
                    if (
                        superclass in allowed_superclasses
                        and str(type_name) not in excluded_types
                    ):
                        relay_set.add(relay_index)
    return tuple(sorted(relay_set))


def _descending_watch(
    setup: Any,
    *,
    max_hops: int,
) -> tuple[int, ...]:
    forward = _effective_forward_distances(
        setup.graph,
        set(setup.lamina_sources),
        max_hops=max_hops,
    )
    table = setup.graph.catalog.table
    return tuple(
        sorted(
            neuron
            for neuron, distance in forward.items()
            if 0 < distance <= max_hops
            and table["superclass"][neuron].as_py() == "descending_neuron"
        )
    )


def _neuron_metadata(
    setup: Any,
    watch_indices: tuple[int, ...],
    *,
    max_hops: int,
) -> list[dict[str, Any]]:
    forward = _effective_forward_distances(
        setup.graph,
        set(setup.lamina_sources),
        max_hops=max_hops,
    )
    table = setup.graph.catalog.table
    rows: list[dict[str, Any]] = []
    for neuron in watch_indices:
        rows.append(
            {
                "neuron_index": neuron,
                "body_id": int(table["body_id"][neuron].as_py()),
                "instance": str(table["instance"][neuron].as_py()),
                "type": str(table["type"][neuron].as_py()),
                "soma_side": str(table["soma_side"][neuron].as_py()),
                "shortest_lamina_hops": int(forward[neuron]),
            }
        )
    return rows


def run_recording(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    motor_config_path = ROOT / str(config["motor_config"])
    motor_config = json.loads(motor_config_path.read_text(encoding="utf-8"))
    base_config_path = ROOT / str(motor_config["base_config"])
    setup = _setup_visual_system(data_directory, base_config_path)

    positions = tuple(float(value) for value in config["positions"])
    if tuple(sorted(positions)) != positions:
        raise ValueError("positions must be sorted")
    max_hops = int(config["maximum_descending_hops"])
    temporal_bins = int(config.get("temporal_bins", 0))
    watch_indices = _descending_watch(setup, max_hops=max_hops)
    if not watch_indices:
        raise RuntimeError("no reachable descending neurons selected")

    relay_indices = _graded_relay_indices(setup, motor_config)
    relay_spec = motor_config.get("graded_relay", {})
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
        graded_relay_indices=relay_indices,
        graded_relay_gain=float(relay_spec.get("gain", 0.0)),
        graded_relay_activation_scale_mv=float(
            relay_spec.get("activation_scale_mv", 7.0)
        ),
    )
    simulator.prepare()

    started = time.perf_counter()
    axes: dict[str, Any] = {}
    for axis in config["axes"]:
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
                temporal_bins=temporal_bins,
            )
            for position in positions
        ]
        axes[str(axis)] = {
            "samples": [
                {
                    "position": float(sample["position"]),
                    "voltage": [
                        float(value) for value in sample["voltage"]
                    ],
                    "drive": [float(value) for value in sample["drive"]],
                    "spike": [float(value) for value in sample["spike"]],
                    "temporal_voltage": [
                        [float(value) for value in row]
                        for row in sample.get("temporal_voltage", [])
                    ],
                    "temporal_drive": [
                        [float(value) for value in row]
                        for row in sample.get("temporal_drive", [])
                    ],
                    "total_network_spikes": int(sample["total_network_spikes"]),
                }
                for sample in samples
            ]
        }

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Neural dynamics are unchanged from the pruned gain=0.25 motor model. "
            "Only the recording watchlist is expanded to all descending neurons "
            "reachable from lamina sources within the configured effective-hop bound."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "motor_config_sha256": file_sha256(motor_config_path),
        "descending_neuron_count": len(watch_indices),
        "graded_relay_count": len(relay_indices),
        "neurons": _neuron_metadata(setup, watch_indices, max_hops=max_hops),
        "axes": axes,
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

    result = run_recording(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "descending_neuron_count": result["descending_neuron_count"],
                "graded_relay_count": result["graded_relay_count"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
