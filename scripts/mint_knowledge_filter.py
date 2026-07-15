"""Folder-level knowledge-base filtering for MINT audit scripts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


DATE_PREFIX_RE = re.compile(r"^(\d{8})")


def _normalize_cell(value: Any) -> str:
    """Normalize Excel cells to stable text for folder matching."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _date_variants(value: Any) -> set[str]:
    """Return exact and 8-digit-prefix variants for a Date folder value."""
    text = _normalize_cell(value)
    if not text:
        return set()
    variants = {text}
    match = DATE_PREFIX_RE.match(text)
    if match:
        variants.add(match.group(1))
    return variants


@dataclass(frozen=True)
class FolderKnowledgeFilter:
    """Allow-list of source folder pairs from an Excel knowledge table.

    The source tree is expected to use:
    ``<ID>/<Date>/<...DICOM files...>``.
    """

    pairs: frozenset[tuple[str, str]]
    source_path: Path
    sheet_name: str
    n_rows: int

    def matches_relative_path(self, relative_path: Path) -> bool:
        """Return True when the path belongs to an allowed ID-Date folder pair."""
        parts = relative_path.parts
        if len(parts) < 2:
            return False
        patient_folder = parts[0]
        date_folder = parts[1]
        return any((patient_folder, variant) in self.pairs for variant in _date_variants(date_folder))


def load_folder_knowledge_filter(
    workbook_path: Path | None,
    sheet_name: str = "RT243-3012-final",
    id_column: str = "ID",
    date_column: str = "Date",
) -> FolderKnowledgeFilter | None:
    """Load ID-Date folder pairs from an Excel workbook.

    Returns ``None`` when no workbook path is provided.
    """
    if workbook_path is None:
        return None
    workbook_path = workbook_path.expanduser().resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Knowledge workbook not found: {workbook_path}")

    table = pd.read_excel(workbook_path, sheet_name=sheet_name)
    missing = [column for column in (id_column, date_column) if column not in table.columns]
    if missing:
        raise ValueError(
            f"Knowledge sheet {sheet_name!r} is missing required column(s): {', '.join(missing)}"
        )

    pairs: set[tuple[str, str]] = set()
    for _, row in table[[id_column, date_column]].dropna(how="any").iterrows():
        patient_id = _normalize_cell(row[id_column])
        if not patient_id:
            continue
        for date_value in _date_variants(row[date_column]):
            pairs.add((patient_id, date_value))

    if not pairs:
        raise ValueError(f"No valid {id_column}-{date_column} pairs found in {workbook_path}")

    return FolderKnowledgeFilter(
        pairs=frozenset(pairs),
        source_path=workbook_path,
        sheet_name=sheet_name,
        n_rows=len(table),
    )
