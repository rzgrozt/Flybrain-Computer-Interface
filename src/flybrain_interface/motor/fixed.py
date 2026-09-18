"""Fixed artificial-body decoder for descending neural populations."""

from __future__ import annotations

from dataclasses import dataclass

from flybrain_interface.contracts import MotorCommand, NeuralReadout


@dataclass(frozen=True, slots=True)
class FixedPopulationMotorDecoder:
    """Map firing-rate differences to motion without learning an action policy."""

    left_population: str = "left"
    right_population: str = "right"
    up_population: str = "up"
    down_population: str = "down"
    click_population: str = "click"
    gain: float = 0.01
    max_delta: float = 1.0
    click_threshold_hz: float = 20.0

    def decode(self, readout: NeuralReadout) -> MotorCommand:
        rates = readout.population_rates_hz
        horizontal = self.gain * (
            rates.get(self.right_population, 0.0)
            - rates.get(self.left_population, 0.0)
        )
        vertical = self.gain * (
            rates.get(self.down_population, 0.0) - rates.get(self.up_population, 0.0)
        )
        return MotorCommand(
            delta_x=self._clamp(horizontal),
            delta_y=self._clamp(vertical),
            click=rates.get(self.click_population, 0.0) >= self.click_threshold_hz,
        )

    def _clamp(self, value: float) -> float:
        return max(-self.max_delta, min(self.max_delta, value))

@dataclass(frozen=True, slots=True)
class ContinuousPopulationMotorDecoder:
    """Map continuous descending-population voltage state to cursor motion.

    Positive horizontal output is screen-right. Positive vertical output is
    screen-down. DNa02-like left/right steering populations therefore use
    right-minus-left, while backward-minus-forward maps locomotor direction onto the
    artificial cursor's vertical axis. Click is intentionally disabled.
    """

    steer_left_population: str = "steer_left"
    steer_right_population: str = "steer_right"
    forward_population: str = "forward"
    backward_population: str = "backward"
    horizontal_gain: float = 1.0
    vertical_gain: float = 1.0
    deadzone_mv: float = 0.0
    max_delta: float = 1.0

    def decode(self, readout: NeuralReadout) -> MotorCommand:
        state = readout.population_voltage_delta_mv
        horizontal_raw = (
            state.get(self.steer_right_population, 0.0)
            - state.get(self.steer_left_population, 0.0)
        )
        vertical_raw = (
            state.get(self.backward_population, 0.0)
            - state.get(self.forward_population, 0.0)
        )
        horizontal = self._apply_deadzone(horizontal_raw) * self.horizontal_gain
        vertical = self._apply_deadzone(vertical_raw) * self.vertical_gain
        return MotorCommand(
            delta_x=self._clamp(horizontal),
            delta_y=self._clamp(vertical),
            click=False,
        )

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) <= self.deadzone_mv else value

    def _clamp(self, value: float) -> float:
        return max(-self.max_delta, min(self.max_delta, value))
