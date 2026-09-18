from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_interface.experiments.early_vision import (
    _graded_summary,
    _stimulus_window_s,
    condition_frames,
    evaluate_qualitative_gates,
    graded_config,
    load_config,
    make_projected_lamina_drive,
)
from flybrain_interface.experiments.visual_pathway import DirectTargetPopulation
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionConfig,
    simulate_graded_early_vision,
    simulate_graded_early_vision_channels,
    simulate_graded_photoreceptor_channels,
)
from flybrain_interface.simulation.config import ShiuLIFConfig


def test_graded_background_has_tonic_release_without_lamina_delta() -> None:
    frames = np.full((6, 2, 2), 0.5, dtype=np.float64)
    trace = simulate_graded_early_vision(frames)

    assert trace.baseline_release > 0.0
    assert np.all(trace.histamine_release > 0.0)
    assert np.max(np.abs(trace.l1_delta_mv)) < 1e-12
    assert np.max(np.abs(trace.l2_delta_mv)) < 1e-12
    assert np.max(np.abs(trace.l3_delta_mv)) < 1e-12


def test_light_and_dark_steps_invert_lamina_response_sign() -> None:
    light = np.broadcast_to(
        np.asarray([0.5, 0.5, 1.0, 1.0, 1.0, 0.5])[:, None, None],
        (6, 2, 2),
    )
    dark = np.broadcast_to(
        np.asarray([0.5, 0.5, 0.0, 0.0, 0.0, 0.5])[:, None, None],
        (6, 2, 2),
    )

    light_trace = simulate_graded_early_vision(light)
    dark_trace = simulate_graded_early_vision(dark)

    for signal in (
        light_trace.l1_delta_mv,
        light_trace.l2_delta_mv,
        light_trace.l3_delta_mv,
    ):
        assert float(np.min(signal)) < 0.0
    for signal in (
        dark_trace.l1_delta_mv,
        dark_trace.l2_delta_mv,
        dark_trace.l3_delta_mv,
    ):
        assert float(np.max(signal)) > 0.0


def test_predeclared_graded_qualitative_gates_pass() -> None:
    config = load_config()
    model = graded_config(config)
    runs = []
    for condition, specification in config["conditions"].items():
        frames = condition_frames(config, condition)
        trace = simulate_graded_early_vision(frames, model)
        start_s, stop_s = _stimulus_window_s(
            specification, float(config["timing"]["frame_rate_hz"])
        )
        runs.append(
            {
                "condition": condition,
                "graded_reference": _graded_summary(trace, start_s, stop_s),
            }
        )

    gates = evaluate_qualitative_gates(runs, config)

    assert gates["passed"]
    assert all(gates["checks"].values())


def test_graded_model_rejects_invalid_physiology_proxy_parameters() -> None:
    with pytest.raises(ValueError, match="fast tau"):
        GradedEarlyVisionConfig(
            transient_fast_tau_ms=50.0,
            transient_slow_tau_ms=40.0,
        )
    with pytest.raises(ValueError, match="normalized"):
        simulate_graded_early_vision(np.full((2, 2, 2), 1.01))


def test_projected_lamina_drive_preserves_delay_and_scaling() -> None:
    config = load_config()
    frames = condition_frames(config, "light_step")
    trace = simulate_graded_early_vision(frames, graded_config(config))
    group = DirectTargetPopulation(
        key="direct_l1",
        label="Direct L1",
        type_name="L1",
        neuron_indices=(1,),
        body_ids=(101,),
        direct_edge_count=1,
        direct_contact_count=3,
        contacts_by_neuron=(3,),
    )
    projection = SimpleNamespace(
        target_indices=np.asarray([2], dtype=np.int64),
        signed_contact_weights=np.asarray([-3.0], dtype=np.float64),
        source_count=1,
        anatomical_edge_count=1,
    )
    lif_config = ShiuLIFConfig(dt_ms=float(config["timing"]["neural_dt_ms"]))

    drive = make_projected_lamina_drive(
        trace,
        (group,),
        {group.key: projection},
        config,
        lif_config,
    )

    channel = drive.channels[0]
    delay_steps = round(
        float(config["projected_drive"]["axonal_delay_ms"]) / lif_config.dt_ms
    )
    expected_scale = (
        lif_config.synapse_scale_mv
        * float(config["projected_drive"]["equivalent_rate_hz"])
        * (lif_config.dt_ms / 1000.0)
    )
    expected_peak = (
        np.max(np.abs(trace.l1_delta_mv))
        / float(config["graded_model"]["l1_gain_mv"])
        * expected_scale
    )

    assert drive.dt_ms == lif_config.dt_ms
    assert np.all(channel.amplitudes_mv[:delay_steps] == 0.0)
    assert np.any(channel.amplitudes_mv[delay_steps:] < 0.0)
    assert np.max(np.abs(channel.amplitudes_mv)) == pytest.approx(expected_peak)


def test_parallel_photoreceptor_channels_keep_background_at_zero_delta() -> None:
    intensity = np.full((6, 3), 0.5, dtype=np.float64)

    trace = simulate_graded_photoreceptor_channels(intensity)

    assert trace.histamine_release_delta.shape[1] == 3
    np.testing.assert_allclose(trace.histamine_release_delta, 0.0, atol=1e-15)
    assert np.all(trace.histamine_release > 0.0)
    assert not trace.histamine_release_delta.flags.writeable


def test_parallel_photoreceptor_channels_preserve_spatial_sign_and_independence(
) -> None:
    intensity = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    trace = simulate_graded_photoreceptor_channels(intensity)

    assert float(np.min(trace.histamine_release_delta[:, 0])) < 0.0
    np.testing.assert_allclose(trace.histamine_release_delta[:, 1], 0.0, atol=1e-15)
    assert float(np.max(trace.histamine_release_delta[:, 2])) > 0.0
    np.testing.assert_allclose(
        trace.histamine_release_delta[:, 0],
        -trace.histamine_release_delta[:, 2],
        atol=1e-14,
    )


def test_parallel_graded_early_vision_matches_scalar_single_channel() -> None:
    luminance = np.asarray([0.5, 0.5, 1.0, 1.0, 0.5, 0.5], dtype=np.float64)
    scalar_frames = np.broadcast_to(luminance[:, None, None], (6, 2, 2)).copy()
    channel_frames = luminance[:, None]

    scalar = simulate_graded_early_vision(scalar_frames)
    parallel = simulate_graded_early_vision_channels(channel_frames)

    np.testing.assert_allclose(
        parallel.photoreceptor.photoreceptor_state[:, 0],
        scalar.photoreceptor_state,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        parallel.photoreceptor.histamine_release[:, 0],
        scalar.histamine_release,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        parallel.l1_delta_mv[:, 0],
        scalar.l1_delta_mv,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        parallel.l2_delta_mv[:, 0],
        scalar.l2_delta_mv,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        parallel.l3_delta_mv[:, 0],
        scalar.l3_delta_mv,
        rtol=0.0,
        atol=1e-14,
    )
