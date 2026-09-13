"""Typed boundaries between the closed-loop system's subsystems.

The contracts deliberately make semantic support incapable of returning motor
commands and make reward evaluators incapable of selecting actions. Only a
``NeuralReadout`` can be accepted by a ``MotorDecoder``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Generic, Protocol, TypeVar

RawSensoryT = TypeVar("RawSensoryT", contravariant=True)
EnvironmentStateT = TypeVar("EnvironmentStateT", contravariant=True)
NeuralInputT = TypeVar("NeuralInputT", contravariant=True)


@dataclass(frozen=True, slots=True)
class StructuredGoal:
    """A semantic goal with no action or coordinate fields."""

    concept: str
    desired_state: str
    qualifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.concept.strip():
            raise ValueError("goal concept cannot be empty")
        if not self.desired_state.strip():
            raise ValueError("goal desired_state cannot be empty")


@dataclass(frozen=True, slots=True)
class SensoryFrame:
    """Synthetic neural input produced by a sensory transducer."""

    timestamp_s: float
    channels: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if any(
            not isfinite(value)
            for values in self.channels.values()
            for value in values
        ):
            raise ValueError("sensory channel values must be finite")


@dataclass(frozen=True, slots=True)
class ImageObservation:
    """Raw RGB observation available only to a sensory or teacher boundary."""

    frame_id: int
    width: int
    height: int
    rgb8: bytes

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.rgb8) != self.width * self.height * 3:
            raise ValueError("rgb8 length must match width * height * 3")


@dataclass(frozen=True, slots=True)
class NeuralReadout:
    """Observable neural output; the sole input accepted by motor decoding."""

    duration_s: float
    neuron_spike_counts: tuple[int, ...]
    population_rates_hz: Mapping[str, float]
    spike_times_s: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if any(count < 0 for count in self.neuron_spike_counts):
            raise ValueError("spike counts cannot be negative")
        if any(
            not isfinite(rate) or rate < 0
            for rate in self.population_rates_hz.values()
        ):
            raise ValueError("population rates must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MotorCommand:
    """Artificial body command produced only by a fixed neural decoder."""

    delta_x: float
    delta_y: float
    click: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.delta_x) or not isfinite(self.delta_y):
            raise ValueError("motor deltas must be finite")


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """Scalar teaching signal without action-policy information."""

    value: float
    terminal: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("reward must be finite")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """A compact observation intended for a non-blocking telemetry sink."""

    topic: str
    timestamp_s: float
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("telemetry topic cannot be empty")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("telemetry timestamp must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ClockRates:
    """Independent clocks; only the neural loop uses a timestep."""

    neural_dt_ms: float = 0.1
    visual_hz: float = 30.0
    motor_hz: float = 25.0
    reward_hz: float = 20.0
    telemetry_hz: float = 10.0
    optional_ai_hz: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.neural_dt_ms,
            self.visual_hz,
            self.motor_hz,
            self.reward_hz,
            self.telemetry_hz,
            self.optional_ai_hz,
        )
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("all clock rates must be finite and positive")


class SensoryEncoder(Protocol, Generic[RawSensoryT]):
    """Translate an external observation into neural sensory channels."""

    def encode(
        self, observation: RawSensoryT, *, timestamp_s: float
    ) -> SensoryFrame: ...


class SemanticTranslator(Protocol):
    """Normalize language into a goal, never an action."""

    def translate(self, instruction: str) -> StructuredGoal: ...


class NeuralSimulator(Protocol, Generic[NeuralInputT]):
    """Transform neural sensory input into neural output."""

    def step(self, sensory: NeuralInputT, *, duration_s: float) -> NeuralReadout: ...


class MotorDecoder(Protocol):
    """Decode only connectome output into an artificial body command."""

    def decode(self, readout: NeuralReadout) -> MotorCommand: ...


class RewardEvaluator(Protocol, Generic[EnvironmentStateT]):
    """Evaluate outcomes and emit only scalar reinforcement."""

    def evaluate(self, state: EnvironmentStateT) -> RewardSignal: ...


class VisualRewardEvaluator(Protocol):
    """Optional visual teacher that can emit reward but never an action."""

    def evaluate(
        self, observation: ImageObservation, goal: StructuredGoal
    ) -> RewardSignal: ...


class TelemetrySink(Protocol):
    """Receive observability data without controlling the simulation."""

    def publish(self, event: TelemetryEvent) -> None: ...
