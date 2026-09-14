"""Resumable, manifest-verified dataset acquisition."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen

from flybrain_interface.connectome_data.manifest import (
    DatasetManifest,
    SourceArtifact,
    VerifiedArtifact,
    verify_artifact,
)


def acquire_dataset(
    manifest: DatasetManifest,
    raw_directory: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[VerifiedArtifact, ...]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    return tuple(
        _acquire_artifact(source, raw_directory, chunk_size)
        for source in manifest.sources
    )


def _acquire_artifact(
    source: SourceArtifact, raw_directory: Path, chunk_size: int
) -> VerifiedArtifact:
    target = raw_directory / source.filename
    if target.exists():
        return verify_artifact(source, raw_directory)

    partial = target.with_name(f"{target.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(source.url, headers=headers)
    with urlopen(request) as response:  # noqa: S310 - URL is manifest-locked
        status = getattr(response, "status", 200)
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(chunk_size):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())

    actual_size = partial.stat().st_size
    if actual_size != source.size_bytes:
        raise ValueError(
            f"incomplete download for {source.filename}: "
            f"{actual_size} != {source.size_bytes}"
        )
    os.replace(partial, target)
    try:
        return verify_artifact(source, raw_directory)
    except Exception:
        target.rename(partial)
        raise
