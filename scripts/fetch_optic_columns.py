"""Fetch the pinned MaleCNS optic-column workbook and verify its lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = (
    ROOT / "data" / "manifests" / "optic-column-type-assignments-v1.0.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "raw" / "optic-column-type-assignments-v1.0.xlsx"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    manifest: dict[str, Any] = json.loads(
        arguments.manifest.read_text(encoding="utf-8")
    )
    url = str(manifest["source_url"])
    expected_size = int(manifest["size_bytes"])
    expected_sha = str(manifest["sha256"])

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
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
                f"downloaded optic-column size mismatch: {size} != {expected_size}"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("downloaded optic-column SHA-256 mismatch")
        temporary.replace(arguments.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "size_bytes": size,
                "sha256": expected_sha,
                "source_url": url,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
