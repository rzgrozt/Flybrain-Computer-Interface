"""Fetch pinned Zhao et al. eye-map source artifacts and verify their locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "eyemap-zhao2025-v1.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data" / "raw" / "eyemap-zhao2025-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    manifest: dict[str, Any] = json.loads(
        arguments.manifest.read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported eye-map source-manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("eye-map source manifest contains no artifacts")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("eye-map source artifact must be an object")
        results.append(
            _fetch_artifact(
                artifact,
                arguments.output_directory,
            )
        )

    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_directory),
                "commit": manifest["commit"],
                "artifacts": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _fetch_artifact(
    artifact: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    filename = str(artifact["filename"])
    url = str(artifact["url"])
    expected_size = int(artifact["size_bytes"])
    expected_sha = str(artifact["sha256"])
    output = output_directory / filename
    temporary = output.with_suffix(output.suffix + ".tmp")

    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open(
            "wb"
        ) as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
                digest.update(block)
                size += len(block)
        if size != expected_size:
            raise ValueError(
                f"downloaded {filename} size mismatch: {size} != {expected_size}"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"downloaded {filename} SHA-256 mismatch")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "filename": filename,
        "size_bytes": size,
        "sha256": expected_sha,
        "url": url,
    }


if __name__ == "__main__":
    main()
