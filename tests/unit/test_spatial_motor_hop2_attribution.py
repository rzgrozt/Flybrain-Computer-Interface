from __future__ import annotations

from flybrain_interface.experiments.spatial_motor_hop2_attribution import (
    _group_status,
    _summarize_groups,
)


def _decoded(*, passes: bool, sides: tuple[float, float, float]) -> dict[str, object]:
    return {
        "passes_generalization": passes,
        "features": {
            "voltage": {"test": {"side_accuracy": sides[0]}},
            "drive": {"test": {"side_accuracy": sides[1]}},
            "spike": {"test": {"side_accuracy": sides[2]}},
        },
    }


def test_group_status_distinguishes_preserve_wrong_side_and_mixed() -> None:
    assert _group_status(_decoded(passes=True, sides=(1.0, 0.5, 0.0))) == "preserve"
    assert _group_status(_decoded(passes=False, sides=(0.0, 0.0, 0.0))) == "wrong-side"
    assert _group_status(_decoded(passes=False, sides=(0.5, 0.0, 0.0))) == "mixed"


def test_group_summary_aggregates_repeated_anatomical_labels() -> None:
    axes = {
        "horizontal": {
            "targets": [
                {
                    "target_instance": "DNa02_L",
                    "groups": {
                        "superclass": [
                            {
                                "label": "visual_projection",
                                "status": "preserve",
                                "best_test_mae": 0.1,
                            }
                        ]
                    },
                }
            ]
        },
        "vertical": {
            "targets": [
                {
                    "target_instance": "DNa02_R",
                    "groups": {
                        "superclass": [
                            {
                                "label": "visual_projection",
                                "status": "mixed",
                                "best_test_mae": 0.3,
                            }
                        ]
                    },
                }
            ]
        },
    }

    summary = _summarize_groups(axes, ("superclass",))
    row = summary["superclass"][0]

    assert row["label"] == "visual_projection"
    assert row["evaluations"] == 2
    assert row["preserve"] == 1
    assert row["mixed"] == 1
    assert row["preserve_fraction"] == 0.5
    assert row["mean_best_test_mae"] == 0.2
    assert row["axes"] == ["horizontal", "vertical"]
