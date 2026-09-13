"""Run a tiny, deterministic Brian2 propagation and decoder benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from time import perf_counter

from flybrain_interface.connectome_data.sparse import SparseConnectivity
from flybrain_interface.motor.fixed import FixedPopulationMotorDecoder
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator


def run_synthetic_reference(runs: int = 1) -> dict[str, object]:
    if runs <= 0:
        raise ValueError("runs must be positive")

    connectivity = SparseConnectivity.from_edges(
        neuron_count=5,
        sources=(0, 1, 2, 2),
        targets=(1, 2, 3, 4),
        signed_synapse_counts=(250.0, 250.0, 250.0, -250.0),
    )
    simulator = Brian2ReferenceSimulator(
        connectivity,
        populations={
            "right": (3,),
            "left": (4,),
            "up": (1,),
            "down": (2,),
            "click": (3,),
        },
    )
    stimulus = DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,))

    started = perf_counter()
    readout = simulator.step(stimulus, duration_s=0.020)
    for _ in range(runs - 1):
        repeated = simulator.step(stimulus, duration_s=0.020)
        if repeated != readout:
            raise RuntimeError("deterministic reference runs diverged")
    elapsed = perf_counter() - started

    command = FixedPopulationMotorDecoder().decode(readout)
    return {
        "scope": "synthetic validation network; not MaleCNS",
        "backend": "Brian2 CPU reference",
        "neurons": connectivity.neuron_count,
        "sparse_edges": connectivity.edge_count,
        "runs": runs,
        "elapsed_s": elapsed,
        "spike_counts": readout.neuron_spike_counts,
        "population_rates_hz": dict(readout.population_rates_hz),
        "fixed_motor_command": asdict(command),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    arguments = parser.parse_args()
    print(json.dumps(run_synthetic_reference(arguments.runs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
