from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from flybrain_interface.experiments.sensory_descending import (
    ResolvedPopulation,
    _verify_deterministic_repeats,
    build_conditions,
    compare_input_patterns,
    load_config,
    make_stimulus,
    resolve_population,
    response_metrics,
)
from flybrain_interface.simulation.trace import ChunkResult


def test_population_selection_is_query_grounded_and_sorted_by_body_id() -> None:
    table = pa.table(
        {
            "neuron_index": [0, 1, 2, 3],
            "body_id": [40, 10, 30, 20],
            "superclass": ["sensory", "sensory", "other", "sensory"],
            "type": ["R", "R", "R", "R"],
        }
    )

    resolved = resolve_population(
        table,
        {
            "key": "test",
            "label": "Test sensory",
            "query": {"superclass": "sensory", "type": "R"},
            "sample_limit": 2,
        },
    )

    assert resolved.available_count == 3
    assert resolved.body_ids == (10, 20)
    assert resolved.neuron_indices == (1, 3)
    assert resolved.query == {"superclass": "sensory", "type": "R"}


def test_predeclared_matrix_has_matched_controls_and_repeats() -> None:
    config = load_config()
    conditions = build_conditions(config)

    assert len(conditions) == 26
    assert [item.condition_id for item in conditions[:2]] == [
        "control-r0",
        "control-r1",
    ]
    keys = {
        (item.input_key, item.amplitude_mv, item.stimulus_duration_s)
        for item in conditions
    }
    assert (None, 0.0, 0.0) in keys
    assert ("r1_r6", 4.0, 0.02) in keys
    assert ("orn_da1", 8.0, 0.05) in keys
    assert ("jo_a1", 8.0, 0.05) in keys
    assert all(
        sum(
            candidate.input_key == input_key
            and candidate.amplitude_mv == amplitude
            and candidate.stimulus_duration_s == duration
            for candidate in conditions
        )
        == 2
        for input_key, amplitude, duration in keys
    )


def test_stimulus_has_same_dose_per_selected_input_neuron() -> None:
    config = load_config()
    condition = next(
        item
        for item in build_conditions(config)
        if item.condition_id == "r1_r6-a8-d20ms-r0"
    )
    population = ResolvedPopulation(
        key="r1_r6",
        label="R1-R6",
        query={"type": "R1-R6"},
        available_count=3,
        neuron_indices=(2, 4, 6),
        body_ids=(20, 40, 60),
        sample_limit=3,
    )

    stimulus = make_stimulus(condition, {"r1_r6": population}, config, 0.1)

    assert len(stimulus.times_s) == 12
    assert stimulus.times_s[:3] == (0.05, 0.05, 0.05)
    assert stimulus.times_s[-3:] == (0.065, 0.065, 0.065)
    assert stimulus.amplitude_mv == 8.0


def test_response_metrics_define_latency_windows_and_no_response() -> None:
    output = ResolvedPopulation(
        key="dn",
        label="DN",
        query={"superclass": "descending_neuron"},
        available_count=2,
        neuron_indices=(3, 4),
        body_ids=(30, 40),
        sample_limit=None,
    )
    result = _chunk_result((3, 3, 4, 4), (0.01, 0.055, 0.09, 0.12))

    metrics = response_metrics(
        result,
        output,
        baseline_end_s=0.05,
        stimulus_end_s=0.10,
        total_s=0.25,
    )

    assert metrics.baseline_spikes == 1
    assert metrics.stimulus_spikes == 2
    assert metrics.post_spikes == 1
    assert metrics.response_latency_s == pytest.approx(0.005)
    assert metrics.response_duration_s == pytest.approx(0.065)
    assert metrics.post_stimulus_persistence_s == pytest.approx(0.02)
    assert not metrics.no_response

    silent = response_metrics(
        _chunk_result((1,), (0.01,)),
        output,
        baseline_end_s=0.05,
        stimulus_end_s=0.10,
        total_s=0.25,
    )
    assert silent.no_response
    assert silent.response_latency_s is None
    assert silent.response_duration_s is None
    assert silent.post_stimulus_persistence_s is None


def test_repeat_verification_ignores_wall_time_but_rejects_result_drift() -> None:
    base = {
        "condition": {
            "input_key": "r1_r6",
            "amplitude_mv": 8.0,
            "stimulus_duration_s": 0.02,
        },
        "total_network_spikes": 4,
        "visited_edges": 12,
        "responses": [{"population": "dn", "total_spikes": 1}],
        "descending_pattern": {
            "body_spike_counts": [{"body_id": 10, "spike_count": 1}]
        },
    }
    repeated = [base, {**base, "wall_seconds": 999.0}]
    _verify_deterministic_repeats(repeated)

    drifted = [base, {**base, "total_network_spikes": 5}]
    with pytest.raises(RuntimeError, match="repeats disagree"):
        _verify_deterministic_repeats(drifted)


def test_between_input_patterns_define_silent_comparisons() -> None:
    def run(input_key: str, counts: list[dict[str, int]]) -> dict[str, object]:
        return {
            "condition": {
                "input_key": input_key,
                "amplitude_mv": 8.0,
                "stimulus_duration_s": 0.02,
                "repeat": 0,
            },
            "descending_pattern": {"body_spike_counts": counts},
        }

    comparisons = compare_input_patterns(
        [
            run("visual", [{"body_id": 1, "spike_count": 2}]),
            run(
                "auditory",
                [
                    {"body_id": 1, "spike_count": 1},
                    {"body_id": 2, "spike_count": 1},
                ],
            ),
            run("silent", []),
        ]  # type: ignore[arg-type]
    )

    visual_auditory = comparisons[0]
    assert visual_auditory["cosine_similarity"] == pytest.approx(2**-0.5)
    assert visual_auditory["active_neuron_jaccard"] == 0.5
    silent = comparisons[-1]
    assert silent["cosine_similarity"] is None
    assert silent["undefined_reason"] == (
        "one or both descending patterns had no spikes"
    )


def _chunk_result(neurons: tuple[int, ...], times: tuple[float, ...]) -> ChunkResult:
    return ChunkResult(
        start_step=0,
        end_step=2500,
        dt_ms=0.1,
        total_spikes=len(neurons),
        population_rates_hz={},
        neuron_spike_counts=None,
        spike_neuron_indices=np.asarray(neurons, dtype=np.int64),
        spike_times_s=np.asarray(times, dtype=np.float64),
        dropped_spike_events=0,
        watched_indices=(),
        sample_times_s=np.arange(2500, dtype=np.float64) * 0.0001,
        voltage_mv=np.empty((2500, 0)),
        synaptic_drive_mv=np.empty((2500, 0)),
    )
