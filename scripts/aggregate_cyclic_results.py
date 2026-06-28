#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


FIELDS = [
    "Instance",
    "Customer group",
    "Repetition",
    "Strategy",
    "Gamma set",
    "Gamma1",
    "Gamma2",
    "Gamma3",
    "Gamma4",
    "Seed",
    "Solver exit code",
    "Initial cost",
    "Improved cost",
    "Worst cost",
    "Mean cost",
    "Elapsed seconds",
    "Feasibility",
    "Truck routes",
    "Drone routes",
    "Validated makespan",
    "Deadline violation",
    "Energy violation",
    "Capacity violation",
    "Full solution",
]


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def read_value(text, label):
    match = re.search(rf"^{re.escape(label)}:[ \t]*(.*?)[ \t]*$", text, re.MULTILINE)
    return match.group(1) if match else ""


def as_number(value, integer=False):
    if value == "":
        return ""
    try:
        return int(value) if integer else float(value)
    except ValueError:
        return ""


def parse_result(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed = read_value(text, "Mean elapsed time").removesuffix(" seconds")
    solution = text.split("Solution Details:\n", 1)[1].strip() if "Solution Details:\n" in text else ""
    truck_match = re.search(r"Truck Routes:\n(.*?)\nDrone Routes:", solution, re.DOTALL)
    drone_match = re.search(r"Drone Routes:\n(.*?)\nTotal validation:", solution, re.DOTALL)
    validation_match = re.search(
        r"Total validation: Makespan=([^,]+), Deadline violation=([^,]+), "
        r"Energy violation=([^,]+), Capacity violation=([^\n]+)",
        solution,
    )
    validation = validation_match.groups() if validation_match else ("", "", "", "")
    return {
        "Instance": read_value(text, "Instance"),
        "Customer group": as_number(read_value(text, "Customer group"), integer=True),
        "Repetition": as_number(read_value(text, "Repetition"), integer=True),
        "Strategy": read_value(text, "Experiment strategy") or read_value(text, "Neighborhood selection"),
        "Gamma set": read_value(text, "Gamma set"),
        "Gamma1": as_number(read_value(text, "Gamma1")),
        "Gamma2": as_number(read_value(text, "Gamma2")),
        "Gamma3": as_number(read_value(text, "Gamma3")),
        "Gamma4": as_number(read_value(text, "Gamma4")),
        "Seed": as_number(read_value(text, "Experiment seed") or read_value(text, "Random seed"), integer=True),
        "Solver exit code": as_number(read_value(text, "Solver exit code"), integer=True),
        "Initial cost": as_number(read_value(text, "Initial solution cost")),
        "Improved cost": as_number(read_value(text, "Improved solution cost")),
        "Worst cost": as_number(read_value(text, "Worst solution cost")),
        "Mean cost": as_number(read_value(text, "Mean solution cost")),
        "Elapsed seconds": as_number(elapsed),
        "Feasibility": read_value(text, "Final solution feasibility"),
        "Truck routes": truck_match.group(1).strip() if truck_match else "",
        "Drone routes": drone_match.group(1).strip() if drone_match else "",
        "Validated makespan": as_number(validation[0]),
        "Deadline violation": as_number(validation[1]),
        "Energy violation": as_number(validation[2]),
        "Capacity violation": as_number(validation[3]),
        "Full solution": solution,
    }


def column_name(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(row_index, column_index, value, style_id=0):
    reference = f"{column_name(column_index)}{row_index}"
    style = f' s="{style_id}"' if style_id else ""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'<c r="{reference}"{style}><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"{style}><is><t>{escape(str(value))}</t></is></c>'


def worksheet_xml(headers, rows, widths):
    wrapped_columns = {
        index for index, header in enumerate(headers, start=1)
        if header in {"Truck routes", "Drone routes", "Full solution"}
    }
    row_xml = []
    for row_index, values in enumerate([headers, *rows], start=1):
        cells = "".join(
            cell_xml(
                row_index,
                column_index,
                value,
                style_id=1 if row_index == 1 else (2 if column_index in wrapped_columns else 0),
            )
            for column_index, value in enumerate(values, start=1)
        )
        row_xml.append(f'<row r="{row_index}">{cells}</row>')
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    last_cell = f"{column_name(len(headers))}{len(rows) + 1}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_cell}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        f'<cols>{columns}</cols><sheetData>'
        + "".join(row_xml)
        + f'</sheetData><autoFilter ref="A1:{last_cell}"/>'
        '</worksheet>'
    )


def write_xlsx(path, detail_rows, summary_rows, detail_sheet_name):
    detail_values = [[row[field] for field in FIELDS] for row in detail_rows]
    summary_headers = [
        "Strategy", "Gamma set", "Gamma1", "Gamma2", "Gamma3", "Gamma4",
        "Customer group", "Runs", "Feasible", "Best cost", "Mean cost", "Worst cost", "Mean elapsed seconds",
    ]
    sheets = [
        (detail_sheet_name, worksheet_xml(
            FIELDS,
            detail_values,
            [18, 15, 12, 12, 12, 10, 10, 10, 10, 10, 16, 15, 15, 15, 15, 17, 14, 55, 55, 20, 20, 18, 20, 80],
        )),
        ("Summary", worksheet_xml(summary_headers, summary_rows, [12, 12, 10, 10, 10, 10, 15, 10, 12, 15, 15, 15, 22])),
    ]
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + '</Types>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        + "".join(
            f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            for index, (name, _) in enumerate(sheets, start=1)
        )
        + '</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>'
        '</styleSheet>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        for index, (_, xml) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_xlsx", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    result_files = sorted(args.results_dir.rglob("*_run_*.txt"), key=lambda path: natural_key(path.name))
    rows = [parse_result(path) for path in result_files]
    rows.sort(key=lambda row: (natural_key(row["Instance"]), row["Repetition"] or 0))

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["Strategy"], row["Gamma set"], row["Gamma1"], row["Gamma2"],
            row["Gamma3"], row["Gamma4"], row["Customer group"],
        )
        grouped[key].append(row)
    summary_rows = []
    for key in sorted(grouped, key=lambda value: tuple(str(part) for part in value)):
        group_rows = grouped[key]
        feasible_costs = [
            row["Improved cost"] for row in group_rows
            if row["Feasibility"] == "FEASIBLE" and isinstance(row["Improved cost"], float)
        ]
        elapsed = [row["Elapsed seconds"] for row in group_rows if isinstance(row["Elapsed seconds"], float)]
        summary_rows.append([
            *key,
            len(group_rows),
            sum(row["Feasibility"] == "FEASIBLE" for row in group_rows),
            min(feasible_costs) if feasible_costs else "",
            statistics.fmean(feasible_costs) if feasible_costs else "",
            max(feasible_costs) if feasible_costs else "",
            statistics.fmean(elapsed) if elapsed else "",
        ])

    strategies = sorted({row["Strategy"] for row in rows if row["Strategy"]})
    detail_sheet_name = f"{strategies[0].capitalize()} results" if len(strategies) == 1 else "Neighborhood results"
    write_xlsx(args.output_xlsx, rows, summary_rows, detail_sheet_name)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Collected {len(rows)} runs into {args.output_xlsx}")


if __name__ == "__main__":
    main()
