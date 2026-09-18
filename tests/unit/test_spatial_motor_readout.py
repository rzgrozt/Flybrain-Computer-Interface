from __future__ import annotations

import pyarrow as pa

from flybrain_interface.experiments.spatial_motor_readout import (
    _channel_state,
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
