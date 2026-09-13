"""Tests for replaceable support, fixed motor, reward, and telemetry boundaries."""

from flybrain_interface.ai_support.providers import (
    OptionalSemanticSupportConfig,
    OptionalVisualSupportConfig,
    SemanticProviderRegistry,
    VisualProviderRegistry,
)
from flybrain_interface.contracts import (
    ImageObservation,
    NeuralReadout,
    RewardSignal,
    StructuredGoal,
    TelemetryEvent,
)
from flybrain_interface.motor.fixed import FixedPopulationMotorDecoder
from flybrain_interface.reward.distance import (
    DistanceImprovementEvaluator,
    DistanceState,
)
from flybrain_interface.sensory.semantic.literal import LiteralGoalTranslator
from flybrain_interface.telemetry.latest import LatestValueTelemetrySink


def test_optional_semantic_support_is_disabled_by_default() -> None:
    registry = SemanticProviderRegistry()

    assert registry.build(OptionalSemanticSupportConfig()) is None


def test_semantic_provider_is_replaceable_without_simulator_dependency() -> None:
    registry = SemanticProviderRegistry()
    registry.register("literal", lambda _: LiteralGoalTranslator(desired_state="open"))

    translator = registry.build(
        OptionalSemanticSupportConfig(enabled=True, provider="literal")
    )

    assert translator is not None
    assert translator.translate("  Spreadsheet   Application ") == StructuredGoal(
        concept="spreadsheet application",
        desired_state="open",
    )


class ConstantVisualEvaluator:
    def evaluate(
        self, observation: ImageObservation, goal: StructuredGoal
    ) -> RewardSignal:
        return RewardSignal(value=float(observation.frame_id), reason=goal.concept)


def test_visual_support_is_disabled_and_replaceable() -> None:
    registry = VisualProviderRegistry()
    assert registry.build(OptionalVisualSupportConfig()) is None
    registry.register("constant", lambda _: ConstantVisualEvaluator())

    evaluator = registry.build(
        OptionalVisualSupportConfig(enabled=True, provider="constant")
    )

    assert evaluator is not None
    reward = evaluator.evaluate(
        ImageObservation(frame_id=2, width=1, height=1, rgb8=b"\x00\x00\x00"),
        StructuredGoal(concept="window open", desired_state="visible"),
    )
    assert reward == RewardSignal(value=2.0, reason="window open")


def test_fixed_motor_decoder_uses_only_neural_rates() -> None:
    readout = NeuralReadout(
        duration_s=0.1,
        neuron_spike_counts=(0, 2),
        population_rates_hz={"left": 2.0, "right": 12.0},
    )

    command = FixedPopulationMotorDecoder(gain=0.1).decode(readout)

    assert command.delta_x == 1.0
    assert command.delta_y == 0.0
    assert command.click is False


def test_distance_evaluator_returns_scalar_improvement() -> None:
    reward = DistanceImprovementEvaluator().evaluate(
        DistanceState(
            previous_cursor=(0.0, 0.0),
            cursor=(1.0, 0.0),
            target=(2.0, 0.0),
        )
    )

    assert reward.value == 1.0
    assert reward.terminal is False


def test_telemetry_drops_obsolete_event() -> None:
    sink = LatestValueTelemetrySink(capacity=1)
    sink.publish(TelemetryEvent(topic="motor", timestamp_s=1.0, values={"x": 1.0}))
    latest = TelemetryEvent(topic="motor", timestamp_s=2.0, values={"x": 2.0})
    sink.publish(latest)

    assert sink.drain_latest() == latest
    assert sink.drain_latest() is None
