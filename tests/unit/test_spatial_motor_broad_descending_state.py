from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa

from flybrain_interface.experiments.spatial_motor_broad_descending_state import (
    _descending_watch,
)


class _Graph:
    def __init__(self) -> None:
        self.catalog = SimpleNamespace(
            table=pa.table(
                {
                    "superclass": [
                        "descending_neuron",
                        "descending_neuron",
                        "central",
                    ]
                }
            )
        )


class _Setup:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.lamina_sources = ()


def test_descending_watch_filters_superclass_and_positive_hops(monkeypatch) -> None:
    setup = _Setup()

    monkeypatch.setattr(
        "flybrain_interface.experiments.spatial_motor_broad_descending_state._effective_forward_distances",
        lambda graph, sources, max_hops: {0: 1, 1: 3, 2: 2},
    )

    assert _descending_watch(setup, max_hops=3) == (0, 1)
