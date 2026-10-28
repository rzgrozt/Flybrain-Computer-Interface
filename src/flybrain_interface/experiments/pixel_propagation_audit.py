"""Locate pixel propagation loss and test real subpaths against Brian2.

Test pulses are explicit interventions, not sensory observations or behaviour.
The canonical neural parameters and fixed motor mapping are never fitted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from flybrain_interface.connectome_data.induced import induce_connectome
from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.validate import validate_processed_dataset
from flybrain_interface.experiments.pixel_fixed_motor import (
    demo_frames,
    normalize_frames,
    run_sequence,
)
from flybrain_interface.experiments.spatial_motor_pathway import (
    _annotation,
    _pathway_records,
    _setup_visual_system,
    _watchlist,
)
from flybrain_interface.experiments.spatial_motor_readout import (
    resolve_motor_populations,
)
from flybrain_interface.experiments.validate_runtime import _maximum_spike_time_error
from flybrain_interface.motor.fixed import FixedPopulationMotorDecoder
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import SimulationTrace

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs/pixel-propagation-audit-v1.json"


def compare_traces(
    expected: SimulationTrace,
    actual: SimulationTrace,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Report exact spike-count and bounded timing/trajectory agreement."""
    spike_error = _maximum_spike_time_error(expected, actual)
    voltage_error = float(np.max(np.abs(expected.voltage_mv - actual.voltage_mv)))
    drive_error = float(
        np.max(np.abs(expected.synaptic_drive_mv - actual.synaptic_drive_mv))
    )
    counts_equal = (
        expected.readout.neuron_spike_counts == actual.readout.neuron_spike_counts
    )
    return {
        "spike_counts_equal": counts_equal,
        "maximum_spike_time_error_s": spike_error if np.isfinite(spike_error) else None,
        "maximum_voltage_error_mv": voltage_error,
        "maximum_drive_error_mv": drive_error,
        "equivalent": bool(
            counts_equal
            and spike_error <= tolerances["spike_time_s"]
            and voltage_error <= tolerances["voltage_mv"]
            and drive_error <= tolerances["drive_mv"]
        ),
    }


def layer_metrics(
    indices: tuple[int, ...], neurons: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Summarize a shortest-path layer; layers across targets may overlap."""
    rows = [neurons[index] for index in indices]
    return {
        "neuron_count": len(rows),
        "max_depolarization_mv": max(row["max_voltage_delta_mv"] for row in rows),
        "max_hyperpolarization_mv": -min(row["min_voltage_delta_mv"] for row in rows),
        "max_abs_drive_mv": max(row["max_abs_drive_mv"] for row in rows),
        "spiking_neuron_count": sum(row["spike_count"] > 0 for row in rows),
        "total_spikes": sum(row["spike_count"] for row in rows),
    }


def run_audit(
    data_directory: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported pixel propagation audit schema")
    if not config["intervention_targets"]:
        raise ValueError("audit requires at least one intervention target")
    pixel_path = ROOT / config["pixel_config"]
    pixel_config = json.loads(pixel_path.read_text())
    verified = validate_processed_dataset(
        ROOT / pixel_config["normalized_data_manifest"], data_directory
    )
    base_path = ROOT / pixel_config["base_config"]
    setup = _setup_visual_system(data_directory, base_path)
    motor_path = ROOT / pixel_config["motor_config"]
    motor_config = json.loads(motor_path.read_text())
    populations = resolve_motor_populations(setup.graph.catalog.table, motor_config)
    anatomy, pathways = _pathway_records(
        setup.graph,
        populations,
        setup.lamina_sources,
        max_hops=int(config["maximum_path_hops"]),
        top_paths=1,
    )
    watch = _watchlist(pathways, setup.lamina_sources)
    if not watch or len(watch) > int(config["maximum_watch_neurons"]):
        raise ValueError("pathway watchlist is empty or exceeds its configured bound")
    mapping = {population.key: population.neuron_indices for population in populations}
    simulator = SparseLIFSimulator(
        setup.graph,
        populations=mapping,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, setup.base["runtime"]["backend"]),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy, setup.base["runtime"]["subnormal_drive_policy"]
        ),
    )
    simulator.prepare()
    decoder = FixedPopulationMotorDecoder(**pixel_config["decoder"])
    frames = normalize_frames(demo_frames(), pixel_config["limits"])
    conditions = {}
    for name, pixels, disconnected in (
        ("pixels", frames, False),
        (
            "blank",
            np.full_like(frames, setup.graded_config.background_luminance),
            False,
        ),
        ("vision_disconnected", frames, True),
    ):
        conditions[name] = run_sequence(
            setup,
            simulator,
            pixels,
            decoder,
            chunk_ms=float(pixel_config["motor_chunk_ms"]),
            disconnected=disconnected,
            extra_watch_indices=watch,
        )
    responses = []
    for target, pathway in pathways.items():
        per_condition = {}
        for name, condition in conditions.items():
            neurons = {row["neuron_index"]: row for row in condition["watched_neurons"]}
            layers = [
                {"hop": hop, **layer_metrics(layer, neurons)}
                for hop, layer in enumerate(pathway.layers[1:], start=1)
            ]
            per_condition[name] = {
                "layers": layers,
                "first_zero_drive_hop": next(
                    (
                        layer["hop"]
                        for layer in layers
                        if layer["max_abs_drive_mv"] <= 1e-12
                    ),
                    None,
                ),
            }
        responses.append(
            {
                "target": _annotation(setup.graph, target),
                "distance": pathway.distance,
                "conditions": per_condition,
            }
        )

    interventions = []
    references = []
    selected_routes = []
    pulse = config["pulse"]
    for instance in config["intervention_targets"]:
        candidates = [
            record for record in anatomy if record["target_instance"] == instance
        ]
        if len(candidates) != 1 or not candidates[0]["reachable"]:
            raise ValueError(
                f"intervention target must resolve to one reachable neuron: {instance}"
            )
        route = candidates[0]["top_paths"][0]
        selected_routes.append({"target_instance": instance, **route})
        # Exclude clamped/externally managed lamina. The reference validates
        # only this induced spiking suffix, not the graded sensory prosthesis.
        original = tuple(node["neuron_index"] for node in route["neurons"][1:])
        if set(original).intersection(setup.clamped):
            raise ValueError("spiking suffix overlaps externally managed visual cells")
        induced = induce_connectome(setup.graph, original)
        for local, neuron in enumerate(original):
            stimulus = DeterministicSpikeInput(
                neuron_indices=(neuron,) * len(pulse["times_s"]),
                times_s=tuple(float(t) for t in pulse["times_s"]),
                amplitude_mv=float(pulse["amplitude_mv"]),
            )
            duration = float(pulse["duration_s"])
            trace = simulator.run(
                stimulus, duration_s=duration, watched_indices=original
            )
            interventions.append(
                {
                    "target_instance": instance,
                    "stimulated_hop": local + 1,
                    "direct_dn_stimulation": local == len(original) - 1,
                    "stimulated_neuron": _annotation(setup.graph, neuron),
                    "scope": "full graph, no pixels, direct voltage test pulses",
                    "pulse": pulse,
                    "network_spikes": sum(trace.readout.neuron_spike_counts),
                    "stimulated_neuron_spikes": trace.readout.neuron_spike_counts[
                        neuron
                    ],
                    "motor_command": asdict(decoder.decode(trace.readout)),
                    "route_neurons": [
                        {
                            "neuron_index": index,
                            "spike_count": trace.readout.neuron_spike_counts[index],
                            "max_voltage_delta_mv": float(
                                np.max(
                                    trace.voltage_mv[:, pos]
                                    - setup.lif_config.resting_mv
                                )
                            ),
                            "min_voltage_delta_mv": float(
                                np.min(
                                    trace.voltage_mv[:, pos]
                                    - setup.lif_config.resting_mv
                                )
                            ),
                            "max_abs_drive_mv": float(
                                np.max(np.abs(trace.synaptic_drive_mv[:, pos]))
                            ),
                        }
                        for pos, index in enumerate(original)
                    ],
                }
            )
            local_stimulus = DeterministicSpikeInput(
                neuron_indices=(local,) * len(stimulus.times_s),
                times_s=stimulus.times_s,
                amplitude_mv=stimulus.amplitude_mv,
            )
            watched_local = tuple(range(induced.neuron_count))
            expected = Brian2ReferenceSimulator(
                induced.as_sparse_connectivity(), config=setup.lif_config
            ).trace(local_stimulus, duration_s=duration, watched_indices=watched_local)
            for backend in ("numpy", "numba"):
                actual = SparseLIFSimulator(
                    induced,
                    config=setup.lif_config,
                    backend=backend,
                ).run(
                    local_stimulus, duration_s=duration, watched_indices=watched_local
                )
                references.append(
                    {
                        "target_instance": instance,
                        "stimulated_hop": local + 1,
                        "backend": backend,
                        "original_indices": original,
                        "induced_edges": induced.edge_count,
                        "reference_spike_counts": expected.readout.neuron_spike_counts,
                        **compare_traces(
                            expected, actual, config["reference_tolerances"]
                        ),
                    }
                )

    source_paths = {
        "audit_config": config_path,
        "pixel_config": pixel_path,
        "base_config": base_path,
        "motor_config": motor_path,
        "audit_source": Path(__file__),
        "pixel_source": ROOT
        / "src/flybrain_interface/experiments/pixel_fixed_motor.py",
        "normalized_manifest": ROOT / pixel_config["normalized_data_manifest"],
        "runtime_source": ROOT / "src/flybrain_interface/simulation/runtime.py",
        "reference_source": ROOT / "src/flybrain_interface/simulation/reference.py",
        "kernel_source": ROOT / "src/flybrain_interface/simulation/kernels.py",
        "retinal_source": ROOT / "src/flybrain_interface/sensory/vision/retinotopic.py",
        "graded_vision_source": ROOT
        / "src/flybrain_interface/sensory/vision/graded.py",
        "lamina_interface_source": ROOT
        / "src/flybrain_interface/experiments/spatial_graded_lamina.py",
        "pathway_source": ROOT
        / "src/flybrain_interface/experiments/spatial_motor_pathway.py",
        "decoder_source": ROOT / "src/flybrain_interface/motor/fixed.py",
    }
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "config": config,
        "scope": (
            "propagation diagnostics; interventions do not demonstrate "
            "perception or learning"
        ),
        "source_hashes": {key: file_sha256(path) for key, path in source_paths.items()},
        "eye_map_lock": setup.eye_map_lock,
        "base_config": setup.base,
        "lif_config": asdict(setup.lif_config),
        "versions": {
            name: version(name)
            for name in ("flybrain-interface", "numpy", "numba", "scipy", "brian2")
        },
        "runtime": {
            "backend": simulator.backend,
            "subnormal_drive_policy": simulator.subnormal_drive_policy,
            "graded_relays_enabled": False,
            "plasticity_enabled": False,
            "os_actions_enabled": False,
        },
        "decoder": asdict(decoder),
        "dataset": asdict(verified),
        "sign_policy": setup.graph.sign_policy.name,
        "pathway_watch_count": len(watch),
        "conditions": conditions,
        "pathway_responses": responses,
        "selected_routes": selected_routes,
        "interventions": interventions,
        "reference_comparisons": references,
        "integrity_checks": {
            "all_targets_reachable": all(record["reachable"] for record in anatomy),
            "no_clamped_spikes": all(
                c["clamped_visual_spikes"] == 0 for c in conditions.values()
            ),
            "blank_silent": conditions["blank"]["total_network_spikes"] == 0,
            "disconnected_silent": conditions["vision_disconnected"][
                "total_network_spikes"
            ]
            == 0,
            "test_pulses_elicit_local_spikes": all(
                item["stimulated_neuron_spikes"] > 0 for item in interventions
            ),
            "direct_dn_pulses_move_fixed_decoder": all(
                item["motor_command"]["delta_x"] != 0
                or item["motor_command"]["delta_y"] != 0
                for item in interventions
                if item["direct_dn_stimulation"]
            ),
            "reference_equivalent": all(item["equivalent"] for item in references),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory", type=Path, default=ROOT / "data/processed/malecns-v1.0"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "watch_count": result["pathway_watch_count"],
                "integrity_checks": result["integrity_checks"],
            },
            indent=2,
        )
    )
    if not all(result["integrity_checks"].values()):
        parser.exit(
            1, "Propagation audit integrity checks failed; see output artifact.\n"
        )


if __name__ == "__main__":
    main()
