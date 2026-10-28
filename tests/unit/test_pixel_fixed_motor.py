"""Pixel-only input, causal motor output and exact chunked-drive semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.connectome_data.runtime import SignedSourceProjection
from flybrain_interface.experiments.pixel_fixed_motor import (
    normalize_frames,
    run_sequence,
    slice_drive,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    CartridgeSources,
    load_config,
)
from flybrain_interface.experiments.spatial_motor_pathway import VisualSetup
from flybrain_interface.experiments.spatial_subthreshold_readout import _graded_config
from flybrain_interface.motor.fixed import FixedPopulationMotorDecoder
from flybrain_interface.sensory.drive import (
    DeterministicProjectedDriveInput,
    ProjectedDriveChannel,
)
from flybrain_interface.sensory.vision import (
    MeasuredColumnDirection,
    VirtualScreenConfig,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import SparseLIFSimulator
from flybrain_interface.simulation.trace import ChunkRecording


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((0, 2, 2)),
        np.zeros((2, 2)),
        np.zeros((1, 2, 2, 4)),
        np.full((1, 2, 2), np.nan),
        np.full((1, 2, 2), 1.1),
        np.zeros((1, 2, 2), dtype=np.int16),
        np.zeros((61, 2, 2)),
        np.zeros((1, 257, 256)),
    ],
)
def test_rejects_invalid_or_unbounded_pixels(frames: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_frames(frames, {"max_frames": 60, "max_pixels_per_frame": 65536})


def test_uint8_is_scaled_without_extracting_target_information() -> None:
    pixels = np.array([[[[0, 127, 255], [255, 0, 64]]]], dtype=np.uint8)
    normalized = normalize_frames(pixels, {"max_frames": 1, "max_pixels_per_frame": 2})
    np.testing.assert_array_equal(normalized, pixels.astype(np.float64) / 255)
    assert normalized.shape == pixels.shape


def _graph() -> InducedConnectome:
    return InducedConnectome(
        original_indices=np.arange(6, dtype=np.int64),
        outgoing_indptr=np.zeros(7, dtype=np.int64),
        target_indices=np.array([], dtype=np.int32),
        synapse_counts=np.array([], dtype=np.int32),
        presynaptic_signs=np.ones(6, dtype=np.int8),
    )


def test_drive_slicing_preserves_exact_voltage_and_spikes_across_motor_ticks() -> None:
    # A drive pulse straddles the chunk boundary. Its accumulated membrane and
    # delayed state must survive that boundary, rather than replaying an onset.
    drive = DeterministicProjectedDriveInput(
        channels=(
            ProjectedDriveChannel(
                label="pulse",
                target_indices=np.array([3]),
                signed_contact_weights=np.array([1.0]),
                amplitudes_mv=np.r_[np.zeros(75), np.full(50, 2.0), np.zeros(75)],
                anatomical_source_count=1,
                anatomical_edge_count=1,
            ),
        ),
        dt_ms=0.1,
    )
    simulator = SparseLIFSimulator(_graph(), backend="numpy")
    recording = ChunkRecording(watched_indices=(3,), include_neuron_counts=True)
    whole = simulator.advance_chunk(
        duration_s=0.02, projected_drive=drive, recording=recording
    )
    simulator.reset()
    first = simulator.advance_chunk(
        duration_s=0.01, projected_drive=slice_drive(drive, 0, 100), recording=recording
    )
    second = simulator.advance_chunk(
        duration_s=0.01,
        projected_drive=slice_drive(drive, 100, 200),
        recording=recording,
    )
    np.testing.assert_array_equal(
        whole.voltage_mv, np.vstack([first.voltage_mv, second.voltage_mv])
    )
    assert whole.total_spikes > 0
    assert whole.total_spikes == first.total_spikes + second.total_spikes
    np.testing.assert_array_equal(
        whole.synaptic_drive_mv,
        np.vstack([first.synaptic_drive_mv, second.synaptic_drive_mv]),
    )


def _synthetic_setup() -> tuple[VisualSetup, SparseLIFSimulator]:
    # Deliberately strong synthetic L2 -> right output for a positive interface
    # control. These weights are not MaleCNS data or a physiological hypothesis.
    column = "ME_R_col_18_19"
    cfg = ShiuLIFConfig()
    base = load_config()
    projections = {
        ("L1", column): SignedSourceProjection(1, 1, np.array([5]), np.array([-1.0])),
        ("L2", column): SignedSourceProjection(
            1, 1, np.array([3]), np.array([10000.0])
        ),
    }
    setup = cast(
        VisualSetup,
        SimpleNamespace(
            graded_config=_graded_config(base, cfg),
            lif_config=cfg,
            base=base,
            directions=(
                MeasuredColumnDirection(
                    side="R",
                    column=column,
                    p=0,
                    q=0,
                    forward=1,
                    right=0,
                    up=0,
                    elevation_deg=0,
                    azimuth_deg=0,
                    source_raw_p=0,
                    source_raw_q=0,
                ),
            ),
            screen=VirtualScreenConfig(120, 90),
            source_columns=(column,),
            cartridges=(CartridgeSources(column, (0,), 1, 2, None),),
            projections=projections,
            clamped=(0, 1, 2),
        ),
    )
    simulator = SparseLIFSimulator(
        _graph(),
        backend="numpy",
        clamped_indices=setup.clamped,
        populations={"left": (4,), "right": (3,), "up": (4,), "down": (4,)},
    )
    return setup, simulator


def test_pixels_drive_fixed_motor_only_through_neural_response() -> None:
    setup, simulator = _synthetic_setup()
    frames = np.full((12, 4, 4, 3), 0.5)
    frames[3:9] = 0
    decoder = FixedPopulationMotorDecoder()
    response = run_sequence(setup, simulator, frames, decoder, chunk_ms=20)
    disconnected = run_sequence(
        setup, simulator, frames, decoder, chunk_ms=20, disconnected=True
    )
    blank = run_sequence(
        setup, simulator, np.full_like(frames, 0.5), decoder, chunk_ms=20
    )
    assert response["motor_nonzero"]
    assert response["clamped_visual_spikes"] == 0
    assert len(response["ticks"]) == 10
    assert response["frame_sha256"] == disconnected["frame_sha256"]
    right = next(row for row in response["watched_neurons"] if row["neuron_index"] == 3)
    assert right["spike_count"] > 0
    assert right["max_abs_drive_mv"] > 0
    for result in (blank, disconnected):
        assert not result["motor_nonzero"]
        assert result["total_network_spikes"] == 0
        assert all(row["max_abs_drive_mv"] == 0 for row in result["watched_neurons"])
    for tick in response["ticks"]:
        assert not tick["motor_command"]["click"]
        assert tick["motor_command"]["delta_x"] == min(
            1.0, tick["population_rates_hz"]["right"] * 0.01
        )


def test_motor_clock_must_align_to_neural_grid() -> None:
    setup, simulator = _synthetic_setup()
    with pytest.raises(ValueError, match="align"):
        run_sequence(
            setup,
            simulator,
            np.zeros((12, 4, 4)),
            FixedPopulationMotorDecoder(),
            chunk_ms=20.05,
        )
