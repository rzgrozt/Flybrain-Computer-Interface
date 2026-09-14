from __future__ import annotations

import pytest

from flybrain_interface.experiments.stability_runtime import (
    RunBudgets,
    _activity_summary,
    _budget_stop_reason,
    _distribution,
    _memory_summary,
)


def test_distribution_reports_interpolated_p95() -> None:
    summary = _distribution([0.4, 0.1, 0.3, 0.2])

    assert summary["count"] == 4
    assert summary["minimum"] == pytest.approx(0.1)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["p95"] == pytest.approx(0.385)
    assert summary["maximum"] == pytest.approx(0.4)


def test_activity_summary_distinguishes_decay_from_end_activity() -> None:
    assert _activity_summary([5, 1, 0, 0], 0.01) == {
        "active_chunk_count": 2,
        "peak_chunk_spikes": 5,
        "last_active_time_seconds": 0.02,
        "final_chunk_spikes": 0,
        "final_ten_percent_spikes": 0,
    }


def test_memory_summary_separates_anonymous_and_file_backed_growth() -> None:
    samples = [
        {
            "rss_bytes": 100,
            "peak_rss_bytes": 110,
            "anonymous_rss_bytes": 40,
            "file_rss_bytes": 60,
            "file_pss_bytes": 50,
            "private_dirty_bytes": 30,
        },
        {
            "rss_bytes": 160,
            "peak_rss_bytes": 170,
            "anonymous_rss_bytes": 50,
            "file_rss_bytes": 110,
            "file_pss_bytes": 90,
            "private_dirty_bytes": 35,
        },
    ]

    assert _memory_summary(samples) == {
        "starting_rss_bytes": 100,
        "ending_rss_bytes": 160,
        "rss_growth_bytes": 60,
        "maximum_rss_bytes": 160,
        "process_peak_rss_bytes_at_end": 170,
        "anonymous_rss_growth_bytes": 10,
        "file_rss_growth_bytes": 50,
        "file_pss_growth_bytes": 40,
        "private_dirty_growth_bytes": 5,
    }


def test_nonfinite_state_precedes_resource_budget_failures() -> None:
    budgets = RunBudgets(wall_seconds=1, rss_bytes=2, spikes=3, visited_edges=4)

    assert _budget_stop_reason(2, 3, 4, 5, True, budgets) == "nonfinite-state"
    assert _budget_stop_reason(2, 1, 1, 1, False, budgets) == "wall-time-budget"
    assert _budget_stop_reason(0, 1, 1, 1, False, budgets) is None
