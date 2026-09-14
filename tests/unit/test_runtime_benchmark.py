from __future__ import annotations

import pytest

from flybrain_interface.experiments.benchmark_runtime import _timing_summary


def test_timing_summary_reports_repeated_range_and_median() -> None:
    assert _timing_summary([0.3, 0.1, 0.2]) == {
        "repeats": 3,
        "minimum_seconds": 0.1,
        "median_seconds": 0.2,
        "maximum_seconds": 0.3,
    }


def test_timing_summary_rejects_empty_measurement() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _timing_summary([])
