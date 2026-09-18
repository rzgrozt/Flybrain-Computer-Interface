from __future__ import annotations

from flybrain_interface.experiments.spatial_motor_tonic import _summarize_bias


def test_tonic_summary_requires_both_motor_axes_and_spike_budget() -> None:
    result = {
        "config": {"tonic_bias_mv": 0.75},
        "total_wall_seconds": 1.0,
        "axes": {
            "horizontal": {
                "runs": [
                    {
                        "channels": {
                            "steering_voltage_right_minus_left_mv": -0.002,
                            "vertical_voltage_backward_minus_forward_mv": 0.0,
                        },
                        "total_network_spikes": 10,
                    },
                    {
                        "channels": {
                            "steering_voltage_right_minus_left_mv": 0.002,
                            "vertical_voltage_backward_minus_forward_mv": 0.0,
                        },
                        "total_network_spikes": 20,
                    },
                ]
            },
            "vertical": {
                "runs": [
                    {
                        "channels": {
                            "steering_voltage_right_minus_left_mv": 0.0,
                            "vertical_voltage_backward_minus_forward_mv": -0.003,
                        },
                        "total_network_spikes": 30,
                    },
                    {
                        "channels": {
                            "steering_voltage_right_minus_left_mv": 0.0,
                            "vertical_voltage_backward_minus_forward_mv": 0.003,
                        },
                        "total_network_spikes": 40,
                    },
                ]
            },
        },
    }
    config = {
        "gates": {
            "minimum_horizontal_steering_range_mv": 0.001,
            "minimum_vertical_forward_backward_range_mv": 0.001,
            "maximum_network_spikes_per_run": 50,
        }
    }

    summary = _summarize_bias(result, config)

    assert summary["horizontal_steering_range_mv"] == 0.004
    assert summary["vertical_forward_backward_range_mv"] == 0.006
    assert summary["max_network_spikes_per_run"] == 40
    assert summary["passed"] is True
