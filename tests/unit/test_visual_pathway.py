from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from flybrain_interface.experiments.visual_pathway import (
    DirectTargetPopulation,
    _direct_response,
    _verify_repeats,
    condition_frames,
    load_config,
    reference_sample,
    resolve_direct_targets,
    resolve_visual_input,
)
from flybrain_interface.sensory.vision import (
    UniformLuminanceConfig,
    encode_uniform_luminance,
)
from flybrain_interface.simulation.trace import ChunkResult


def test_uniform_encoder_is_deterministic_and_frame_clock_is_independent() -> None:
    frames = np.asarray([np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 0.5)])
    config = UniformLuminanceConfig(
        frame_rate_hz=60.0,
        max_pulses_per_frame=4,
        pulse_amplitude_mv=8.0,
        neural_dt_ms=0.1,
        start_s=0.05,
    )

    first = encode_uniform_luminance(frames, (2, 4), config)
    second = encode_uniform_luminance(frames.copy(), (2, 4), config)

    assert first == second
    assert first.frame_timestamps_s[1] == pytest.approx(1 / 60 + 0.05)
    assert first.frame_timestamps_s[1] / 0.0001 != pytest.approx(
        round(first.frame_timestamps_s[1] / 0.0001)
    )
    assert all(
        time / 0.0001 == pytest.approx(round(time / 0.0001))
        for time in first.stimulus.times_s
    )
    assert first.pulse_counts_per_frame == (0, 4, 2)
    assert len(first.stimulus.times_s) == 12


def test_static_and_step_controls_have_equal_engineered_dose() -> None:
    config = load_config()
    encoder_config = UniformLuminanceConfig(
        frame_rate_hz=config["timing"]["frame_rate_hz"],
        max_pulses_per_frame=config["transduction"]["max_pulses_per_frame"],
        pulse_amplitude_mv=config["transduction"]["pulse_amplitude_mv"],
        neural_dt_ms=0.1,
        start_s=config["timing"]["baseline_s"],
    )

    no_input = encode_uniform_luminance(
        condition_frames(config, "no_input"), (1,), encoder_config
    )
    static = encode_uniform_luminance(
        condition_frames(config, "static_half"), (1,), encoder_config
    )
    step = encode_uniform_luminance(
        condition_frames(config, "dark_to_bright"), (1,), encoder_config
    )

    assert len(no_input.stimulus.times_s) == 0
    assert len(static.stimulus.times_s) == len(step.stimulus.times_s) == 12
    assert static.pulse_counts_per_frame == (2, 2, 2, 2, 2, 2)
    assert step.pulse_counts_per_frame == (0, 0, 0, 4, 4, 4)


def test_encoder_rejects_non_normalized_images_and_colliding_density() -> None:
    config = UniformLuminanceConfig(60.0, 4, 8.0, 0.1)
    with pytest.raises(ValueError, match="normalized"):
        encode_uniform_luminance(np.full((1, 2, 2), 1.1), (1,), config)
    with pytest.raises(ValueError, match="distinct neural steps"):
        UniformLuminanceConfig(1000.0, 11, 8.0, 0.1)


def test_visual_selection_uses_all_matching_r1_r6_and_reference_is_prefix() -> None:
    table = pa.table(
        {
            "neuron_index": [0, 1, 2, 3],
            "body_id": [40, 10, 30, 20],
            "superclass": ["ol_sensory", "ol_sensory", "other", "ol_sensory"],
            "class": ["visual", "visual", "visual", "visual"],
            "type": ["R1-R6", "R1-R6", "R1-R6", "R1-R6"],
        }
    )
    config = load_config()

    population = resolve_visual_input(table, config)
    reference = reference_sample(population, 2)

    assert population.body_ids == (10, 20, 40)
    assert population.neuron_indices == (1, 3, 0)
    assert reference.body_ids == (10, 20)
    assert reference.available_count == 3


def test_direct_targets_are_graph_grounded_and_body_sorted() -> None:
    table = pa.table(
        {
            "neuron_index": [0, 1, 2, 3],
            "body_id": [100, 200, 40, 30],
            "type": ["R1-R6", "R1-R6", "L1", "L1"],
        }
    )
    catalog = SimpleNamespace(
        table=table,
        body_ids=lambda indices: np.asarray(table["body_id"])[indices],
    )
    graph = SimpleNamespace(
        catalog=catalog,
        outgoing_indptr=np.asarray([0, 2, 3, 3, 3], dtype=np.int64),
        target_indices=np.asarray([2, 3, 3], dtype=np.int32),
        outgoing_synapse_counts=np.asarray([5, 7, 11], dtype=np.int32),
    )

    result = resolve_direct_targets(graph, (0, 1), "L1")  # type: ignore[arg-type]

    assert result.body_ids == (30, 40)
    assert result.neuron_indices == (3, 2)
    assert result.contacts_by_neuron == (18, 5)
    assert result.direct_edge_count == 3
    assert result.direct_contact_count == 23


def test_subthreshold_response_distinguishes_hyperpolarization_from_spikes() -> None:
    result = ChunkResult(
        start_step=0,
        end_step=4,
        dt_ms=0.1,
        total_spikes=0,
        population_rates_hz={},
        neuron_spike_counts=np.zeros(4, dtype=np.int64),
        spike_neuron_indices=np.empty(0, dtype=np.int64),
        spike_times_s=np.empty(0, dtype=np.float64),
        dropped_spike_events=0,
        watched_indices=(2, 3),
        sample_times_s=np.asarray([0.05, 0.051, 0.052, 0.053]),
        voltage_mv=np.asarray(
            [[-52.0, -52.0], [-52.02, -52.0], [-52.2, -52.1], [-52.0, -52.0]]
        ),
        synaptic_drive_mv=np.asarray(
            [[0.0, 0.0], [-1.0, 0.0], [-0.5, -0.25], [0.0, 0.0]]
        ),
    )
    population = DirectTargetPopulation(
        key="direct_l1",
        label="Direct L1",
        type_name="L1",
        neuron_indices=(2, 3),
        body_ids=(20, 30),
        direct_edge_count=2,
        direct_contact_count=10,
        contacts_by_neuron=(5, 5),
    )

    response = _direct_response(
        result,
        population,
        {2: 0, 3: 1},
        baseline_s=0.05,
        stimulus_end_s=0.052,
        threshold_mv=0.01,
        resting_mv=-52.0,
    )

    assert response["spikes"] == 0
    assert response["responsive_subthreshold_neuron_count"] == 2
    assert response["subthreshold_latency_s"] == pytest.approx(0.001)
    assert response["subthreshold_persistence_s"] == 0.0
    assert response["no_spike_response"]
    assert not response["no_subthreshold_response"]


def test_repeat_check_ignores_resource_variation_but_rejects_neural_drift() -> None:
    base = {"condition": "static", "repeat": 0, "input_spikes": 4}
    _verify_repeats(
        [
            {**base, "wall_seconds": 1.0, "process_rss_bytes": 10},
            {**base, "repeat": 1, "wall_seconds": 2.0, "process_rss_bytes": 20},
        ]
    )
    with pytest.raises(RuntimeError, match="repeats disagree"):
        _verify_repeats([base, {**base, "repeat": 1, "input_spikes": 5}])
