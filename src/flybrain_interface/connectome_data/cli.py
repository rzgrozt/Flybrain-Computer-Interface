"""Command-line interface for reproducible MaleCNS data preparation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flybrain_interface.connectome_data.acquire import acquire_dataset
from flybrain_interface.connectome_data.manifest import load_manifest, verify_dataset
from flybrain_interface.connectome_data.normalize import normalize_dataset
from flybrain_interface.connectome_data.validate import validate_processed_dataset

DEFAULT_MANIFEST = Path("data/manifests/malecns-v1.0.json")
DEFAULT_RAW = Path("data/raw/malecns-v1.0")
DEFAULT_OUTPUT = Path("data/processed/malecns-v1.0")
DEFAULT_LOCK = Path("data/manifests/malecns-v1.0-normalized-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "download", help="download missing files and verify all hashes"
    )
    subparsers.add_parser("verify", help="verify local files against the manifest")
    build = subparsers.add_parser("build", help="normalize verified inputs")
    build.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--memory-limit", default="4GB")
    build.add_argument("--threads", type=int, default=4)
    validate = subparsers.add_parser("validate", help="validate normalized outputs")
    validate.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    validate.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    if arguments.command == "download":
        artifacts = acquire_dataset(manifest, arguments.raw_directory)
        _print_json([asdict(artifact) for artifact in artifacts])
    elif arguments.command == "verify":
        artifacts = verify_dataset(manifest, arguments.raw_directory)
        _print_json([asdict(artifact) for artifact in artifacts])
    elif arguments.command == "build":
        report = normalize_dataset(
            manifest,
            arguments.raw_directory,
            arguments.output_directory,
            memory_limit=arguments.memory_limit,
            threads=arguments.threads,
        )
        _print_json(asdict(report))
    else:
        validation = validate_processed_dataset(
            arguments.lock, arguments.output_directory
        )
        _print_json(asdict(validation))


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
