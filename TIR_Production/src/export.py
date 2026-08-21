#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.export
# Writes the TIRs classified during a session out to a single Excel workbook.

import io
from datetime import datetime
from typing import Dict, List, Sequence
import pandas as pd

from src.config import TARGETS, heading


# The identifying fields a coder can type alongside the description, in the
# order they belong in the sheet.
DETAIL_FIELDS: List[str] = [
    "doc_number", "item_id", "item_type", "severity",
    "doc_title", "description_1", "description_2",
]


# Builds the sheet from the rows accumulated during a session.
def build_frame(rows: Sequence[Dict]) -> pd.DataFrame:
    """Return one row per classified TIR, in reading order.

    Identifying fields first, then the predicted categories, then the
    confidence and review flags a coder needs in order to decide which rows
    still want a second look.
    """
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        record: Dict[str, object] = {
            heading(field): row.get(field, "") for field in DETAIL_FIELDS
        }

        for target in TARGETS:
            record[heading(TARGETS[target].get("column", target))] = row.get(
                f"pred_{target}", ""
            )
            alternates = row.get(f"alt_{target}")
            if alternates:
                record[f"{heading(TARGETS[target].get('column', target))} — other options"] = alternates

        for target in TARGETS:
            title = heading(TARGETS[target].get("column", target))
            record[f"{title} confidence"] = round(float(row.get(f"confidence_{target}", 0.0)), 4)
            record[f"{title} review?"] = "Yes" if row.get(f"review_{target}") else ""

        # Where a language model supplied the code it also supplied its
        # reasoning, and that reasoning is the point of having asked: a coder
        # checking a suggestion needs to see the argument for it, not just the
        # code.  Only the fields that actually carry one get a column, since
        # the model is consulted on few rows and forty empty columns help
        # nobody.
        for target in TARGETS:
            reason = str(row.get(f"rationale_{target}", "") or "").strip()
            if reason:
                title = heading(TARGETS[target].get("column", target))
                record[f"{title} — why"] = reason

        record["Source"] = row.get("source", "model")
        record["Classified At"] = row.get("classified_at", "")
        records.append(record)

    return pd.DataFrame(records)


# Renders the workbook to bytes for a download button.
def to_workbook(rows: Sequence[Dict]) -> bytes:
    """Return the session's predictions as a formatted .xlsx file.

    Written to a buffer rather than a path so the web app can hand it straight
    to a download button without a temporary file.
    """
    frame = build_frame(rows)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="TIR Predictions", index=False)
        sheet = writer.sheets["TIR Predictions"]

        # Freeze the headings so they stay put while a coder scrolls, and bold
        # them so the sheet reads as a report rather than a dump.
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = cell.font.copy(bold=True)

        for index, column in enumerate(frame.columns, start=1):
            longest = max(
                [len(str(column))] + [len(str(v)) for v in frame[column].head(200)]
            )
            sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = (
                min(max(longest + 2, 10), 60)
            )
            if str(column).endswith("confidence"):
                for row_index in range(2, len(frame) + 2):
                    sheet.cell(row=row_index, column=index).number_format = "0.0%"

    return buffer.getvalue()


# Names the downloaded file after the moment it was produced.
def filename(now: datetime | None = None) -> str:
    """Return the download filename, stamped so successive exports don't clash."""
    return f"tir_predictions_{(now or datetime.now()).strftime('%Y%m%d_%H%M')}.xlsx"
