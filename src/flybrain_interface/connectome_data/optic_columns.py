"""Official MaleCNS optic-column supplemental data without Excel dependencies."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS = {"m": _MAIN_NS}
_COLUMN_RE = re.compile(r"^ME_([LR])_col_(\d+)_(\d+)$")


@dataclass(frozen=True, slots=True)
class OpticColumnRecord:
    """One official medulla column row.

    grid_row and grid_column are labels parsed from the official column name.
    They are not screen coordinates and do not imply visual-field orientation.
    """

    column: str
    medulla_side: str
    grid_row: int
    grid_column: int
    l1_body_id: int | None
    r7_body_id: int | None
    r8_body_id: int | None
    column_type: str | None


def load_optic_columns(path: Path) -> tuple[OpticColumnRecord, ...]:
    """Read the official right/left optic-lobe worksheets."""

    sheets = _read_data_sheets(path)
    records: list[OpticColumnRecord] = []
    seen_columns: set[str] = set()
    seen_l1: set[int] = set()
    for sheet_name, rows in sheets:
        if not rows:
            raise ValueError(f"optic-column worksheet is empty: {sheet_name}")
        header = [_normalize_header(value) for value in rows[0]]
        required = {"column", "l1", "r7", "r8"}
        missing = required - set(header)
        if missing:
            raise ValueError(
                f"optic-column worksheet {sheet_name!r} missing fields: "
                f"{sorted(missing)}"
            )
        positions = {name: header.index(name) for name in required}
        type_position = next(
            (
                index
                for index, name in enumerate(header)
                if name in {"column_type", "columntype", "type"}
            ),
            None,
        )
        for row_number, row in enumerate(rows[1:], start=2):
            column_value = _cell(row, positions["column"])
            if column_value in (None, ""):
                continue
            column = str(column_value).strip()
            match = _COLUMN_RE.fullmatch(column)
            if match is None:
                raise ValueError(
                    f"unsupported optic-column label in {sheet_name} row "
                    f"{row_number}: {column!r}"
                )
            if column in seen_columns:
                raise ValueError(f"duplicate optic-column label: {column}")
            seen_columns.add(column)

            l1 = _optional_body_id(_cell(row, positions["l1"]), row_number, "L1")
            r7 = _optional_body_id(_cell(row, positions["r7"]), row_number, "R7")
            r8 = _optional_body_id(_cell(row, positions["r8"]), row_number, "R8")
            if l1 is not None:
                if l1 in seen_l1:
                    raise ValueError(
                        f"duplicate L1 body ID in optic-column table: {l1}"
                    )
                seen_l1.add(l1)
            column_type = None
            if type_position is not None:
                value = _cell(row, type_position)
                if value not in (None, ""):
                    column_type = str(value).strip()

            records.append(
                OpticColumnRecord(
                    column=column,
                    medulla_side=match.group(1),
                    grid_row=int(match.group(2)),
                    grid_column=int(match.group(3)),
                    l1_body_id=l1,
                    r7_body_id=r7,
                    r8_body_id=r8,
                    column_type=column_type,
                )
            )
    if not records:
        raise ValueError("optic-column workbook contains no data rows")
    return tuple(records)


def _read_data_sheets(
    path: Path,
) -> tuple[tuple[str, list[list[str | float | int | None]]], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        shared = _shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_targets = {
            relation.get("Id"): relation.get("Target")
            for relation in relationships
            if relation.get("Id") and relation.get("Target")
        }
        sheets: list[tuple[str, list[list[str | float | int | None]]]] = []
        relationship_key = (
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        for sheet in workbook.findall(".//m:sheets/m:sheet", _NS):
            name = sheet.get("name")
            relation_id = sheet.get(relationship_key)
            if not name or not relation_id:
                raise ValueError("workbook sheet is missing name or relationship ID")
            if name not in {"Right OL", "Left OL"}:
                continue
            target = relationship_targets.get(relation_id)
            if target is None:
                raise ValueError(f"worksheet relationship not found: {relation_id}")
            worksheet_path = f"xl/{target.lstrip('/')}"
            worksheet = ET.fromstring(archive.read(worksheet_path))
            sheets.append((name, _parse_sheet_rows(worksheet, shared)))
    if {name for name, _ in sheets} != {"Right OL", "Left OL"}:
        raise ValueError("optic-column workbook must contain Right OL and Left OL")
    return tuple(sheets)


def _parse_sheet_rows(
    worksheet: ET.Element,
    shared: tuple[str, ...],
) -> list[list[str | float | int | None]]:
    parsed: list[list[str | float | int | None]] = []
    for row in worksheet.findall(".//m:sheetData/m:row", _NS):
        values: dict[int, str | float | int | None] = {}
        maximum = -1
        for cell in row.findall("m:c", _NS):
            reference = cell.get("r")
            if not reference:
                raise ValueError("worksheet cell is missing its coordinate")
            column_index = _column_index(reference)
            maximum = max(maximum, column_index)
            values[column_index] = _cell_value(cell, shared)
        if maximum < 0:
            parsed.append([])
        else:
            parsed.append([values.get(index) for index in range(maximum + 1)])
    return parsed


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ()
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return tuple(
        "".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t"))
        for item in root.findall("m:si", _NS)
    )


def _cell_value(
    cell: ET.Element,
    shared: tuple[str, ...],
) -> str | float | int | None:
    value = cell.find("m:v", _NS)
    if value is None or value.text is None:
        inline = cell.find("m:is", _NS)
        if inline is None:
            return None
        return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))
    raw = value.text
    kind = cell.get("t")
    if kind == "s":
        return shared[int(raw)]
    if kind in {"str", "inlineStr"}:
        return raw
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    integer = int(numeric)
    return integer if numeric == integer else numeric


def _column_index(reference: str) -> int:
    letters = ""
    for character in reference:
        if character.isalpha():
            letters += character
        else:
            break
    if not letters:
        raise ValueError(f"invalid worksheet coordinate: {reference}")
    result = 0
    for character in letters:
        result = result * 26 + (ord(character.upper()) - ord("A") + 1)
    return result - 1


def _normalize_header(value: str | float | int | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _optional_body_id(
    value: str | float | int | None,
    row_number: int,
    label: str,
) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {label} body ID at row {row_number}: {value!r}"
        ) from error
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"non-integer {label} body ID at row {row_number}: {value!r}")
    if integer < 0:
        return None
    return integer


def _cell(
    row: list[str | float | int | None],
    position: int,
) -> str | float | int | None:
    return row[position] if position < len(row) else None
