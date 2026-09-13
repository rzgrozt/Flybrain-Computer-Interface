"""Replaceable optional semantic-provider registration.

No model implementation is installed here. Providers can only satisfy the narrow
``SemanticTranslator`` contract and are disabled unless configuration enables one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from flybrain_interface.contracts import SemanticTranslator, VisualRewardEvaluator

SemanticProviderFactory = Callable[[Mapping[str, object]], SemanticTranslator]
VisualProviderFactory = Callable[[Mapping[str, object]], VisualRewardEvaluator]


@dataclass(frozen=True, slots=True)
class OptionalSemanticSupportConfig:
    enabled: bool = False
    provider: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OptionalVisualSupportConfig:
    enabled: bool = False
    provider: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)


class SemanticProviderRegistry:
    """Build a selected provider without coupling it to neural simulation."""

    def __init__(self) -> None:
        self._factories: dict[str, SemanticProviderFactory] = {}

    def register(self, name: str, factory: SemanticProviderFactory) -> None:
        if not name:
            raise ValueError("provider name cannot be empty")
        if name in self._factories:
            raise ValueError(f"semantic provider already registered: {name}")
        self._factories[name] = factory

    def build(
        self, config: OptionalSemanticSupportConfig
    ) -> SemanticTranslator | None:
        if not config.enabled:
            return None
        if config.provider is None:
            raise ValueError("enabled semantic support requires a provider")
        try:
            factory = self._factories[config.provider]
        except KeyError as error:
            raise ValueError(
                f"unknown semantic provider: {config.provider}"
            ) from error
        return factory(config.options)


class VisualProviderRegistry:
    """Build a visual reward provider without exposing motor interfaces."""

    def __init__(self) -> None:
        self._factories: dict[str, VisualProviderFactory] = {}

    def register(self, name: str, factory: VisualProviderFactory) -> None:
        if not name:
            raise ValueError("provider name cannot be empty")
        if name in self._factories:
            raise ValueError(f"visual provider already registered: {name}")
        self._factories[name] = factory

    def build(
        self, config: OptionalVisualSupportConfig
    ) -> VisualRewardEvaluator | None:
        if not config.enabled:
            return None
        if config.provider is None:
            raise ValueError("enabled visual support requires a provider")
        try:
            factory = self._factories[config.provider]
        except KeyError as error:
            raise ValueError(f"unknown visual provider: {config.provider}") from error
        return factory(config.options)
