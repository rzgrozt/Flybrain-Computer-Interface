from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flybrain_interface.connectome_data.optic_columns import load_optic_columns


def test_optic_column_parser_reads_both_eyes_and_missing_ids(tmp_path: Path) -> None:
    workbook = tmp_path / "optic-columns.xlsx"
    _write_minimal_workbook(workbook)

    records = load_optic_columns(workbook)

    assert len(records) == 2
    right, left = records
    assert right.column == "ME_R_col_10_06"
    assert right.medulla_side == "R"
    assert (right.grid_row, right.grid_column) == (10, 6)
    assert right.l1_body_id == 39722
    assert right.r7_body_id is None
    assert right.r8_body_id == 123
    assert right.column_type == "pale"

    assert left.column == "ME_L_col_11_07"
    assert left.medulla_side == "L"
    assert left.l1_body_id == 50001
    assert left.r7_body_id == 50002
    assert left.r8_body_id is None
    assert left.column_type == "yellow1"


def _write_minimal_workbook(path: Path) -> None:
    workbook_xml = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Right OL" sheetId="1" r:id="rId1"/>
    <sheet name="Left OL" sheetId="2" r:id="rId2"/>
    <sheet name="READ ME" sheetId="3" r:id="rId3"/>
  </sheets>
</workbook>
"""
    relationships_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="worksheets/sheet1.xml"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
  <Relationship Id="rId2" Target="worksheets/sheet2.xml"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
  <Relationship Id="rId3" Target="worksheets/sheet3.xml"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
</Relationships>
"""
    right_sheet = _sheet_xml(
        ["column", "L1", "R7", "R8", "column_type"],
        ["ME_R_col_10_06", 39722, -99, 123, "pale"],
    )
    left_sheet = _sheet_xml(
        ["column", "L1", "R7", "R8", "column_type"],
        ["ME_L_col_11_07", 50001, 50002, -99, "yellow1"],
    )
    readme_sheet = _sheet_xml(["field", "description"], ["column", "column roi"])
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        archive.writestr("xl/worksheets/sheet1.xml", right_sheet)
        archive.writestr("xl/worksheets/sheet2.xml", left_sheet)
        archive.writestr("xl/worksheets/sheet3.xml", readme_sheet)


def _sheet_xml(header: list[object], row: list[object]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main"><sheetData>'
        f'<row r="1">{_cells(header, 1)}</row>'
        f'<row r="2">{_cells(row, 2)}</row>'
        "</sheetData></worksheet>"
    )


def _cells(values: list[object], row: int) -> str:
    cells = []
    for index, value in enumerate(values):
        reference = f"{chr(ord('A') + index)}{row}"
        if isinstance(value, str):
            cells.append(
                f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'
            )
        else:
            cells.append(f'<c r="{reference}"><v>{value}</v></c>')
    return "".join(cells)
