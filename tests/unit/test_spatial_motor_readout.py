from __future__ import annotations

import numpy as np
import pyarrow as pa

from flybrain_interface.experiments.spatial_motor_readout import (
    MotorPopulation,
    _channel_state,
    _population_metrics,
    resolve_motor_population,
)


def test_resolve_motor_population_uses_type_and_soma_side() -> None:
    table = pa.table(
        {
            "neuron_index": [10, 11, 12],
            "body_id": [300, 100, 200],
            "superclass": [
                "descending_neuron",
                "descending_neuron",
                "descending_neuron",
            ],
            "type": ["DNa02", "DNa02", "DNa01"],
            "soma_side": ["R", "L", "L"],
            "instance": ["DNa02_R", "DNa02_L", "DNa01_L"],
        }
    )

    population = resolve_motor_population(
        table,
        {"key": "steer_left", "type": "DNa02", "soma_side": "L"},
    )

    assert population.neuron_indices == (11,)
    assert population.body_ids == (100,)
    assert population.instances == ("DNa02_L",)


def test_channel_state_uses_anatomical_motor_differences() -> None:
    def metrics(voltage: float, drive: float) -> dict[str, float]:
        return {
            "mean_signed_peak_voltage_delta_mv": voltage,
            "mean_signed_peak_drive_mv": drive,
        }

    channels = _channel_state(
        {
            "steer_left": metrics(0.2, 0.3),
            "steer_right": metrics(0.7, 0.8),
            "steer_left_secondary": metrics(0.1, 0.0),
            "steer_right_secondary": metrics(0.4, 0.0),
            "forward": metrics(0.6, 0.5),
            "backward": metrics(0.1, 0.2),
        }
    )

    assert abs(channels["steering_voltage_right_minus_left_mv"] - 0.5) < 1e-12
    assert abs(channels["vertical_voltage_backward_minus_forward_mv"] + 0.5) < 1e-12


def test_population_metrics_preserve_member_level_motor_state() -> None:
    population = MotorPopulation(
        key="backward",
        type_name="MDN",
        soma_side=None,
        neuron_indices=(10, 11),
        body_ids=(100, 101),
        instances=("MDN_R", "MDN_L"),
    )
    sample = {
        "voltage": np.asarray([0.4, -0.2]),
        "drive": np.asarray([0.7, -0.3]),
        "spike": np.asarray([2.0, 1.0]),
    }

    result = _population_metrics(sample, population, {10: 0, 11: 1})

    assert result["spike_count"] == 3
    assert result["members"] == [
        {
            "neuron_index": 10,
            "body_id": 100,
            "instance": "MDN_R",
            "signed_peak_voltage_delta_mv": 0.4,
            "signed_peak_drive_mv": 0.7,
            "spike_count": 2,
        },
        {
            "neuron_index": 11,
            "body_id": 101,
            "instance": "MDN_L",
            "signed_peak_voltage_delta_mv": -0.2,
            "signed_peak_drive_mv": -0.3,
            "spike_count": 1,
        },
    ]
