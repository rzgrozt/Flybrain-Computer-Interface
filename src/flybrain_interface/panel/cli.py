"""Launch the local-only laboratory panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from flybrain_interface.panel.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
        help="read-only normalized MaleCNS directory",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/panel"),
        help="ignored local experiment output directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the first panel release binds to loopback only")
    uvicorn.run(
        create_app(arguments.data_directory, arguments.output_directory),
        host=arguments.host,
        port=arguments.port,
        log_level="info",
    )
