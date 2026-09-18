from __future__ import annotations

import numpy as np

from flybrain_interface.connectome_data.runtime import SignedSourceProjection
from flybrain_interface.experiments.spatial_topographic_readout import (
    activity_centroid,
    backproject_response,
    build_anatomical_backprojection,
    fit_affine,
    pooled_profile,
    predict_affine,
)


def _projection(
    targets: list[int],
    weights: list[float],
) -> SignedSourceProjection:
    return SignedSourceProjection(
        source_count=1,
        anatomical_edge_count=len(targets),
        target_indices=np.asarray(targets, dtype=np.int64),
        signed_contact_weights=np.asarray(weights, dtype=np.float64),
    )


def test_backprojection_uses_absolute_contacts_and_normalizes_each_target() -> None:
    mapping = build_anatomical_backprojection(
        {
            ("L1", "left"): _projection([10, 11], [-3.0, -1.0]),
            ("L2", "right"): _projection([10, 11], [1.0, 3.0]),
        },
        (10, 11),
        {
            "left": (0.2, 0.4),
            "right": (0.8, 0.6),
        },
    )

    np.testing.assert_allclose(
        mapping.target_to_column,
        np.asarray([[0.75, 0.25], [0.25, 0.75]]),
    )
    np.testing.assert_allclose(mapping.target_contact_weight, [4.0, 4.0])


def test_backprojected_centroid_tracks_active_target_topography() -> None:
    mapping = build_anatomical_backprojection(
        {
            ("L1", "left"): _projection([10], [-4.0]),
            ("L1", "right"): _projection([11], [-4.0]),
        },
        (10, 11),
        {
            "left": (0.2, 0.3),
            "right": (0.8, 0.7),
        },
    )

    left_activity = backproject_response(
        np.asarray([3.0, 0.0]),
        mapping,
        epsilon=1e-12,
    )
    right_activity = backproject_response(
        np.asarray([0.0, -5.0]),
        mapping,
        epsilon=1e-12,
    )

    assert activity_centroid(left_activity, mapping.u) == 0.2
    assert activity_centroid(right_activity, mapping.u) == 0.8


def test_pooled_profile_is_low_dimensional_and_normalized() -> None:
    activity = np.asarray([0.1, 0.2, 0.3, 0.4])
    coordinates = np.asarray([0.1, 0.3, 0.6, 0.9])
    profile = pooled_profile(activity, coordinates, bin_count=4)

    np.testing.assert_allclose(profile, [0.1, 0.2, 0.3, 0.4])
    assert np.isclose(np.sum(profile), 1.0)


def test_affine_calibration_generalizes_linear_centroid() -> None:
    train_centroid = np.asarray([0.2, 0.35, 0.5, 0.65, 0.8])
    train_position = np.asarray([0.15, 0.325, 0.5, 0.675, 0.85])
    test_centroid = np.asarray([0.275, 0.425, 0.575, 0.725])
    expected = np.asarray([0.2375, 0.4125, 0.5875, 0.7625])

    model = fit_affine(train_centroid, train_position)
    predicted = predict_affine(model, test_centroid)

    np.testing.assert_allclose(predicted, expected, atol=1e-12)
