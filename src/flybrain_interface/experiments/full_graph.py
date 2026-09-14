"""Inspect and benchmark the normalized full MaleCNS graph."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--active-neurons", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.active_neurons <= 0:
        parser.error("--active-neurons must be positive")

    rss_before = _maximum_rss_bytes()
    started = time.perf_counter()
    graph = MemoryMappedConnectome.load(arguments.data_directory)
    load_seconds = time.perf_counter() - started
    rss_after_load = _maximum_rss_bytes()

    rng = np.random.default_rng(arguments.seed)
    active_count = min(arguments.active_neurons, graph.neuron_count)
    active = rng.choice(graph.neuron_count, size=active_count, replace=False)
    activity = np.zeros(graph.neuron_count, dtype=np.float64)
    activity[active] = 1.0
    started = time.perf_counter()
    postsynaptic_input = graph.propagate(activity)
    propagation_seconds = time.perf_counter() - started
    event_input = np.zeros(graph.neuron_count, dtype=np.float64)
    started = time.perf_counter()
    visited_edges = graph.accumulate_spikes(active, event_input)
    event_seconds = time.perf_counter() - started
    equivalent = bool(np.array_equal(event_input, postsynaptic_input))
    if not equivalent:
        raise RuntimeError("event-driven and full-matrix propagation disagree")

    print(
        json.dumps(
            {
                "data_directory": str(graph.directory),
                "neuron_count": graph.neuron_count,
                "edge_count": graph.edge_count,
                "mapped_edge_bytes": graph.mapped_edge_bytes,
                "load_seconds": load_seconds,
                "max_rss_increase_bytes": max(0, rss_after_load - rss_before),
                "sign_policy": graph.sign_policy.name,
                "sign_counts": graph.sign_counts(),
                "transmitter_counts": graph.transmitter_counts(),
                "active_neurons": active_count,
                "propagation_seconds": propagation_seconds,
                "event_propagation_seconds": event_seconds,
                "event_speedup": propagation_seconds / event_seconds,
                "event_visited_edges": visited_edges,
                "event_matches_full_propagation": equivalent,
                "nonzero_target_inputs": int(np.count_nonzero(postsynaptic_input)),
                "absolute_contact_input": float(np.abs(postsynaptic_input).sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _maximum_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this project currently targets Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


if __name__ == "__main__":
    main()
