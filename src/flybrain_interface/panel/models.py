"""Validated, bounded experiment protocol for the local panel."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StimulusConfig(BaseModel):
    """Declared artificial voltage pulses delivered on the neural time grid."""

    model_config = ConfigDict(extra="forbid")
    neuron_indices: list[int] = Field(default_factory=lambda: [0], max_length=4096)
    start_s: float = Field(default=0.0, ge=0.0, le=60.0)
    stop_s: float = Field(default=0.05, gt=0.0, le=60.0)
    interval_ms: float = Field(default=5.0, ge=0.1, le=10_000.0)
    amplitude_mv: float = Field(default=8.0, gt=0.0, le=100.0)

    @field_validator("neuron_indices")
    @classmethod
    def unique_targets(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("at least one stimulus target is required")
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("stimulus neuron indices must be unique and non-negative")
        return values

    @model_validator(mode="after")
    def valid_window(self) -> StimulusConfig:
        if self.stop_s <= self.start_s:
            raise ValueError("stimulus stop_s must be greater than start_s")
        return self


class PopulationConfig(BaseModel):
    """Engineering observation group; no biological function is implied."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9 _-]+$")
    neuron_indices: list[int] = Field(min_length=1, max_length=4096)

    @field_validator("neuron_indices")
    @classmethod
    def unique_members(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("population indices must be unique and non-negative")
        return values


class ExperimentConfig(BaseModel):
    """Bounded configuration accepted by the isolated simulation worker."""

    model_config = ConfigDict(extra="forbid")
    duration_s: float = Field(default=1.0, ge=0.01, le=60.0)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    backend: Literal["numpy", "numba"] = "numba"
    subnormal_drive_policy: Literal["preserve", "zero"] = "preserve"
    chunk_duration_s: float = Field(default=0.02, ge=0.001, le=0.25)
    telemetry_hz: float = Field(default=10.0, ge=1.0, le=30.0)
    watch_indices: list[int] = Field(default_factory=lambda: [0], max_length=32)
    visualization_indices: list[int] = Field(
        default_factory=lambda: [0], max_length=4096
    )
    populations: list[PopulationConfig] = Field(
        default_factory=lambda: [
            PopulationConfig(name="stimulus targets", neuron_indices=[0])
        ],
        max_length=16,
    )
    stimulus: StimulusConfig = Field(default_factory=StimulusConfig)

    @field_validator("watch_indices")
    @classmethod
    def unique_watchlist(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("watch indices must be unique and non-negative")
        return values

    @field_validator("visualization_indices")
    @classmethod
    def unique_visualization_indices(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("visualization indices must be unique and non-negative")
        return values

    @model_validator(mode="after")
    def aligned_bounds(self) -> ExperimentConfig:
        if self.chunk_duration_s > self.duration_s:
            raise ValueError("chunk duration cannot exceed experiment duration")
        if self.stimulus.stop_s > self.duration_s:
            raise ValueError("stimulus window cannot exceed experiment duration")
        names = [population.name for population in self.populations]
        if len(names) != len(set(names)):
            raise ValueError("population names must be unique")
        return self


class AnatomyMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    neuron_indices: list[int] = Field(max_length=4096)

    @field_validator("neuron_indices")
    @classmethod
    def unique_indices(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("neuron indices must be unique and non-negative")
        return values


class NeighborhoodRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_indices: list[int] = Field(min_length=1, max_length=64)
    max_nodes: int = Field(default=128, ge=2, le=256)
    max_edges: int = Field(default=512, ge=1, le=2048)
    min_synapse_count: int = Field(default=5, ge=1, le=2**31 - 1)

    @field_validator("seed_indices")
    @classmethod
    def unique_seeds(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("seed indices must be unique and non-negative")
        return values


class ActionResponse(BaseModel):
    accepted: bool
    status: str
    detail: str
