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
