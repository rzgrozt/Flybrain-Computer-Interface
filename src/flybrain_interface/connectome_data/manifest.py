"""Versioned source manifests and integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    role: str
    filename: str
    url: str
    size_bytes: int
    gcs_generation: str
    gcs_md5_base64: str
    gcs_crc32c_base64: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.role or not self.filename or not self.url:
            raise ValueError("source role, filename, and URL are required")
        if PurePath(self.filename).name != self.filename:
            raise ValueError("source filename must not contain a path")
        if self.size_bytes <= 0:
            raise ValueError("source size must be positive")
        if len(self.sha256) != 64:
            raise ValueError("source SHA-256 must have 64 hexadecimal characters")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("source SHA-256 must be hexadecimal") from error


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    description: str
    required_superclass: bool
    expected_neuron_count: int

    def __post_init__(self) -> None:
        if not self.required_superclass:
            raise ValueError("schema v1 requires superclass-based neuron selection")
        if self.expected_neuron_count <= 0:
            raise ValueError("expected neuron count must be positive")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset: str
    dataset_uuid: str
    release_date: str
    license: str
    landing_page: str
    selection: SelectionPolicy
    sources: tuple[SourceArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        roles = [source.role for source in self.sources]
        if len(roles) != len(set(roles)):
            raise ValueError("source roles must be unique")
        required = {"annotations", "neurotransmitters", "connectivity"}
        if set(roles) != required:
            raise ValueError(
                f"manifest source roles must be exactly {sorted(required)}"
            )

    def source(self, role: str) -> SourceArtifact:
        try:
            return next(source for source in self.sources if source.role == role)
        except StopIteration as error:
            raise KeyError(role) from error


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    role: str
    path: Path
    size_bytes: int
    sha256: str


def load_manifest(path: Path) -> DatasetManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    selection = SelectionPolicy(**payload.pop("selection"))
    sources = tuple(SourceArtifact(**source) for source in payload.pop("sources"))
    return DatasetManifest(selection=selection, sources=sources, **payload)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(source: SourceArtifact, raw_directory: Path) -> VerifiedArtifact:
    path = raw_directory / source.filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != source.size_bytes:
        raise ValueError(
            f"size mismatch for {source.filename}: {actual_size} != {source.size_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != source.sha256:
        raise ValueError(f"SHA-256 mismatch for {source.filename}")
    return VerifiedArtifact(
        role=source.role,
        path=path,
        size_bytes=actual_size,
        sha256=actual_sha256,
    )


def verify_dataset(
    manifest: DatasetManifest, raw_directory: Path
) -> tuple[VerifiedArtifact, ...]:
    return tuple(verify_artifact(source, raw_directory) for source in manifest.sources)
