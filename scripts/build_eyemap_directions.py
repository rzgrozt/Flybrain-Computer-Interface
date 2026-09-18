"""Build a local measured MaleCNS column-direction table from pinned eye-map data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import load_optic_columns
from flybrain_interface.sensory.vision.retinotopic import (
    MeasuredColumnDirection,
    column_name_from_pq,
    column_pq,
)

ROOT = Path(__file__).parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "manifests" / "eyemap-zhao2025-v1.json"
DEFAULT_SOURCE_DIRECTORY = ROOT / "data" / "raw" / "eyemap-zhao2025-v1"
DEFAULT_OPTIC_WORKBOOK = (
    ROOT / "data" / "raw" / "optic-column-type-assignments-v1.0.xlsx"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "eyemap-view-directions-v1.csv"
DEFAULT_METADATA = ROOT / "data" / "processed" / "eyemap-view-directions-v1.json"

_R_EXTRACT = r"""
args <- commandArgs(trailingOnly=TRUE)
source_dir <- args[1]
raw_output <- args[2]
anchor_output <- args[3]

load(file.path(source_dir, "20240701.RData"))
load(file.path(source_dir, "20240701_nb.RData"))

extract_eye <- function(side, mask, ind_xy) {
  directions <- ucl_rot_sm[order(i_match),][mask,,drop=FALSE]
  selected <- directions[ind_xy[,1],,drop=FALSE]
  norms <- sqrt(rowSums(selected^2))
  selected <- selected / norms
  data.frame(
    side=side,
    raw_p=as.integer(ind_xy[,2]),
    raw_q=as.integer(ind_xy[,3]),
    x=selected[,1],
    y=selected[,2],
    z=selected[,3]
  )
}

raw <- rbind(
  extract_eye("R", !ind_left_lens, ind_xy_right),
  extract_eye("L", ind_left_lens, ind_xy_left)
)
write.csv(raw, raw_output, row.names=FALSE, quote=FALSE)

load(file.path(source_dir, "eyemap.RData"))
manual_directions <- ucl_rot_sm
manual_eyemap <- eyemap
load(file.path(source_dir, "med_ixy.RData"))
anchor_index <- match(manual_eyemap[,1], med_ixy[,1])
if (any(is.na(anchor_index))) {
  stop("manual eye-map Mi1 indices are missing from med_ixy")
}
anchor <- data.frame(
  p=as.integer(med_ixy[anchor_index,2]),
  q=as.integer(med_ixy[anchor_index,3]),
  x=manual_directions[,1],
  y=manual_directions[,2],
  z=manual_directions[,3]
)
write.csv(anchor, anchor_output, row.names=FALSE, quote=FALSE)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument(
        "--optic-workbook",
        type=Path,
        default=DEFAULT_OPTIC_WORKBOOK,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    arguments = parser.parse_args()

    source_manifest = _verify_sources(
        arguments.source_manifest,
        arguments.source_directory,
    )
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is required only to build the local measured-direction table"
        )

    with tempfile.TemporaryDirectory(prefix="flybrain-eyemap-") as directory:
        temporary_directory = Path(directory)
        r_file = temporary_directory / "extract.R"
        raw_csv = temporary_directory / "raw-directions.csv"
        anchor_csv = temporary_directory / "right-anchors.csv"
        r_file.write_text(_R_EXTRACT, encoding="utf-8")
        subprocess.run(
            [
                rscript,
                str(r_file),
                str(arguments.source_directory),
                str(raw_csv),
                str(anchor_csv),
            ],
            check=True,
        )
        raw_rows = _load_vectors(raw_csv, include_side=True)
        anchors = _load_vectors(anchor_csv, include_side=False)

    right_anchor_metrics = _validate_right_anchors(raw_rows, anchors)
    bilateral_metrics = _validate_bilateral_symmetry(raw_rows)
    records, coverage = _join_official_columns(
        raw_rows,
        arguments.optic_workbook,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    _write_records(records, output_tmp)
    output_sha = file_sha256(output_tmp)
    output_tmp.replace(arguments.output)

    metadata = {
        "schema_version": 1,
        "resource": "measured MaleCNS optic-column viewing directions",
        "interpretation": (
            "Local derived table from pinned Zhao et al. 2025 microCT viewing "
            "directions; only exact MaleCNS column-coordinate matches are retained."
        ),
        "source_manifest": str(arguments.source_manifest),
        "source_manifest_sha256": file_sha256(arguments.source_manifest),
        "source_commit": source_manifest["commit"],
        "source_license": source_manifest["license"],
        "output": str(arguments.output),
        "output_sha256": output_sha,
        "record_count": len(records),
        "coverage": coverage,
        "right_anchor_validation": right_anchor_metrics,
        "bilateral_validation": bilateral_metrics,
        "coordinate_convention": source_manifest["coordinate_convention"],
        "runtime_requires_r": False,
        "build_requires_rscript": True,
        "primary_data_vendored": False,
    }
    arguments.metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata_tmp = arguments.metadata.with_suffix(arguments.metadata.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_tmp.replace(arguments.metadata)

    print(json.dumps(metadata, indent=2, sort_keys=True))


def _verify_sources(
    manifest_path: Path,
    source_directory: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported eye-map source-manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("eye-map source manifest contains no artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("eye-map source artifact must be an object")
        path = source_directory / str(artifact["filename"])
        if not path.is_file():
            raise FileNotFoundError(
                f"missing pinned eye-map artifact: {path}; "
                "run scripts/fetch_eyemap_primary.py"
            )
        size = path.stat().st_size
        expected_size = int(artifact["size_bytes"])
        if size != expected_size:
            raise ValueError(
                f"eye-map artifact size mismatch for {path.name}: "
                f"{size} != {expected_size}"
            )
        digest = file_sha256(path)
        if digest != str(artifact["sha256"]):
            raise ValueError(f"eye-map artifact SHA-256 mismatch: {path.name}")
    return manifest


def _load_vectors(
    path: Path,
    *,
    include_side: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, Any] = {
                "p": int(row["raw_p"] if include_side else row["p"]),
                "q": int(row["raw_q"] if include_side else row["q"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
            if include_side:
                parsed["side"] = str(row["side"])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"extracted eye-map table is empty: {path}")
    return rows


def _validate_right_anchors(
    raw_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    right = {
        (int(row["p"]) + 1, int(row["q"])): row
        for row in raw_rows
        if row["side"] == "R"
    }
    if len(right) != sum(row["side"] == "R" for row in raw_rows):
        raise ValueError("duplicate right-eye raw hex coordinates")

    maximum_error = 0.0
    missing = 0
    for anchor in anchors:
        key = (int(anchor["p"]), int(anchor["q"]))
        raw = right.get(key)
        if raw is None:
            missing += 1
            continue
        error = math.sqrt(
            (float(raw["x"]) - float(anchor["x"])) ** 2
            + (float(raw["y"]) - float(anchor["y"])) ** 2
            + (float(raw["z"]) - float(anchor["z"])) ** 2
        )
        maximum_error = max(maximum_error, error)
    if len(anchors) != 778:
        raise RuntimeError(f"unexpected right-eye anchor count: {len(anchors)}")
    if missing:
        raise RuntimeError(f"right-eye coordinate transform misses {missing} anchors")
    if maximum_error > 1e-12:
        raise RuntimeError(
            f"right-eye raw/manual direction mismatch: {maximum_error}"
        )
    return {
        "anchor_count": len(anchors),
        "raw_to_malecns_shift_p": 1,
        "raw_to_malecns_shift_q": 0,
        "missing_anchor_count": missing,
        "maximum_direction_vector_error": maximum_error,
        "passed": True,
    }


def _validate_bilateral_symmetry(
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    right = {
        (int(row["p"]) + 1, int(row["q"])): row
        for row in raw_rows
        if row["side"] == "R"
    }
    left = {
        (int(row["p"]), int(row["q"])): row
        for row in raw_rows
        if row["side"] == "L"
    }
    common = sorted(set(right) & set(left))
    if len(common) < 800:
        raise RuntimeError("insufficient bilateral eye-map coordinate overlap")

    angles: list[float] = []
    for key in common:
        right_row = right[key]
        left_row = left[key]
        rx, ry, rz = (
            float(right_row["x"]),
            float(right_row["y"]),
            float(right_row["z"]),
        )
        lx, ly, lz = (
            float(left_row["x"]),
            float(left_row["y"]),
            float(left_row["z"]),
        )
        right_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        left_norm = math.sqrt(lx * lx + ly * ly + lz * lz)
        cosine = (rx * lx + ry * (-ly) + rz * lz) / (
            right_norm * left_norm
        )
        cosine = max(-1.0, min(1.0, cosine))
        angles.append(math.degrees(math.acos(cosine)))
    sorted_angles = sorted(angles)
    mean_angle = sum(angles) / len(angles)
    median_angle = sorted_angles[len(sorted_angles) // 2]
    maximum_angle = max(angles)
    if mean_angle > 3.0 or median_angle > 3.0 or maximum_angle > 10.0:
        raise RuntimeError(
            "left-eye identity p/q alignment fails bilateral mirror validation"
        )
    return {
        "common_coordinate_count": len(common),
        "left_raw_to_malecns_shift_p": 0,
        "left_raw_to_malecns_shift_q": 0,
        "mean_mirror_angle_error_deg": mean_angle,
        "median_mirror_angle_error_deg": median_angle,
        "maximum_mirror_angle_error_deg": maximum_angle,
        "passed": True,
    }


def _join_official_columns(
    raw_rows: list[dict[str, Any]],
    workbook_path: Path,
) -> tuple[list[MeasuredColumnDirection], dict[str, Any]]:
    columns = load_optic_columns(workbook_path)
    official = {
        (column.medulla_side, *column_pq(column)): column for column in columns
    }
    records: list[MeasuredColumnDirection] = []
    raw_counts = Counter(str(row["side"]) for row in raw_rows)
    outside_counts: Counter[str] = Counter()

    for row in raw_rows:
        side = str(row["side"])
        raw_p = int(row["p"])
        raw_q = int(row["q"])
        p = raw_p + 1 if side == "R" else raw_p
        q = raw_q
        key = (side, p, q)
        column = official.get(key)
        if column is None:
            outside_counts[side] += 1
            continue

        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        norm = math.sqrt(x * x + y * y + z * z)
        forward = x / norm
        right = -y / norm
        up = z / norm
        records.append(
            MeasuredColumnDirection(
                side=side,
                column=column_name_from_pq(side, p, q),
                p=p,
                q=q,
                forward=forward,
                right=right,
                up=up,
                elevation_deg=math.degrees(math.asin(up)),
                azimuth_deg=math.degrees(math.atan2(right, forward)),
                source_raw_p=raw_p,
                source_raw_q=raw_q,
            )
        )

    seen = {record.column for record in records}
    if len(seen) != len(records):
        raise RuntimeError("measured direction join produced duplicate columns")

    official_counts = Counter(column.medulla_side for column in columns)
    retained_counts = Counter(record.side for record in records)
    coverage = {
        side: {
            "official_column_count": official_counts[side],
            "raw_measured_count": raw_counts[side],
            "retained_exact_match_count": retained_counts[side],
            "raw_outside_official_count": outside_counts[side],
            "official_without_measured_direction_count": (
                official_counts[side] - retained_counts[side]
            ),
            "official_coverage_fraction": (
                retained_counts[side] / official_counts[side]
            ),
        }
        for side in ("L", "R")
    }
    coverage["combined"] = {
        "official_column_count": sum(official_counts.values()),
        "retained_exact_match_count": len(records),
        "official_coverage_fraction": len(records) / sum(official_counts.values()),
    }
    records.sort(key=lambda record: (record.side, record.q, record.p))
    return records, coverage


def _write_records(
    records: list[MeasuredColumnDirection],
    path: Path,
) -> None:
    fields = [
        "side",
        "column",
        "p",
        "q",
        "forward",
        "right",
        "up",
        "elevation_deg",
        "azimuth_deg",
        "source_raw_p",
        "source_raw_q",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "side": record.side,
                    "column": record.column,
                    "p": record.p,
                    "q": record.q,
                    "forward": f"{record.forward:.17g}",
                    "right": f"{record.right:.17g}",
                    "up": f"{record.up:.17g}",
                    "elevation_deg": f"{record.elevation_deg:.17g}",
                    "azimuth_deg": f"{record.azimuth_deg:.17g}",
                    "source_raw_p": record.source_raw_p,
                    "source_raw_q": record.source_raw_q,
                }
            )


if __name__ == "__main__":
    main()
