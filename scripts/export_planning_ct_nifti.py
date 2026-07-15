#!/usr/bin/env python3
"""Export manually resolved planning CT series to NIfTI.

The script reads ``mint_manual_review_final_planning_ct_resolved.csv`` and exports
one NIfTI file per unique planning CT. When ``final_ct_series_uid`` is available,
all matching CT DICOM files under the patient folder are aggregated by
SeriesInstanceUID. This handles MINT cases where one CT series is split across
multiple source folders.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd
import pydicom
import SimpleITK as sitk


DATE_PREFIX_RE = re.compile(r"^(\d{8})")


@dataclass(frozen=True)
class PlanningCtCase:
    """One unique planning CT export target."""

    patient_id: str
    final_ct_relative_path: str
    final_ct_series_uid: str
    expected_slices: int | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Read-only MINT raw data root")
    parser.add_argument("--resolved-csv", type=Path, required=True, help="Resolved planning CT CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for NIfTI outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of unique CT cases to export")
    parser.add_argument("--dry-run", action="store_true", help="Print planned exports without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI/QC outputs")
    parser.add_argument(
        "--qc-name",
        default="planningCT_export_qc.csv",
        help="QC CSV filename written inside --output-dir",
    )
    return parser.parse_args()


def missing(value: Any) -> bool:
    """Return True for empty, NaN, or string 'nan' values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return not text or text.lower() == "nan"


def safe_int(value: Any) -> int | None:
    """Parse an integer-like CSV value."""
    if missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def date_from_relative_path(relative_path: str) -> str:
    """Extract the 8-digit examination date from the first path component."""
    first_part = Path(relative_path).parts[0] if Path(relative_path).parts else ""
    match = DATE_PREFIX_RE.match(first_part)
    if not match:
        raise ValueError(f"Could not parse 8-digit date from final_ct_relative_path: {relative_path}")
    return match.group(1)


def load_cases(resolved_csv: Path, limit: int | None) -> list[PlanningCtCase]:
    """Load unique planning CT export cases from the resolved CSV."""
    table = pd.read_csv(resolved_csv)
    required = {"patient_id", "final_ct_relative_path", "final_ct_series_uid", "final_number_of_ct_slices"}
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(f"Resolved CSV missing required columns: {', '.join(missing_columns)}")

    cases: list[PlanningCtCase] = []
    seen: set[tuple[str, str, str]] = set()
    for row in table.itertuples(index=False):
        patient_id = str(getattr(row, "patient_id")).strip()
        relative_path = str(getattr(row, "final_ct_relative_path")).strip()
        series_uid_value = getattr(row, "final_ct_series_uid")
        series_uid = "" if missing(series_uid_value) else str(series_uid_value).strip()
        key = (patient_id, relative_path, series_uid)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            PlanningCtCase(
                patient_id=patient_id,
                final_ct_relative_path=relative_path,
                final_ct_series_uid=series_uid,
                expected_slices=safe_int(getattr(row, "final_number_of_ct_slices")),
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def sort_key(item: tuple[Path, pydicom.dataset.FileDataset]) -> tuple[float, int, str]:
    """Sort CT slices by ImagePositionPatient z, then InstanceNumber."""
    path, ds = item
    image_position = getattr(ds, "ImagePositionPatient", None)
    if image_position is not None and len(image_position) >= 3:
        try:
            z_position = float(image_position[2])
            instance = int(getattr(ds, "InstanceNumber", 0) or 0)
            return z_position, instance, str(path)
        except (TypeError, ValueError):
            pass
    try:
        instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    except (TypeError, ValueError):
        instance = 0
    return 0.0, instance, str(path)


def collect_ct_files(source_dir: Path, case: PlanningCtCase) -> tuple[list[tuple[Path, pydicom.dataset.FileDataset]], dict[str, int], set[str]]:
    """Collect matching CT DICOM files for a planning CT case."""
    patient_root = source_dir / case.patient_id
    listed_folder = patient_root / case.final_ct_relative_path
    scan_root = patient_root if case.final_ct_series_uid else listed_folder
    if not scan_root.exists():
        raise FileNotFoundError(f"Input CT scan root not found: {scan_root}")

    items: list[tuple[Path, pydicom.dataset.FileDataset]] = []
    folder_counts: dict[str, int] = {}
    transfer_syntaxes: set[str] = set()
    for path in scan_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue
        if case.final_ct_series_uid and str(getattr(ds, "SeriesInstanceUID", "")) != case.final_ct_series_uid:
            continue
        items.append((path, ds))
        try:
            relative_parent = str(path.parent.relative_to(patient_root))
        except ValueError:
            relative_parent = str(path.parent)
        folder_counts[relative_parent] = folder_counts.get(relative_parent, 0) + 1
        transfer_syntaxes.add(str(getattr(ds.file_meta, "TransferSyntaxUID", "")))
    return items, folder_counts, transfer_syntaxes


def export_case(source_dir: Path, output_dir: Path, case: PlanningCtCase, overwrite: bool) -> dict[str, Any]:
    """Export a single planning CT case to NIfTI and return a QC row."""
    exam_date = date_from_relative_path(case.final_ct_relative_path)
    output_path = output_dir / f"{case.patient_id}_{exam_date}_planningCT.nii.gz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use --overwrite to replace: {output_path}")

    items, folder_counts, transfer_syntaxes = collect_ct_files(source_dir, case)
    if not items:
        raise ValueError(f"No matching CT DICOM files found for patient {case.patient_id}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path, _ in sorted(items, key=sort_key)])
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path), True)

    nifti_shape = ""
    voxel_size = ""
    warning: list[str] = []
    image_3d = nib.load(str(output_path))
    nifti_shape = "x".join(str(value) for value in image_3d.shape)
    voxel_size = "x".join(f"{value:g}" for value in image_3d.header.get_zooms()[:3])
    if case.expected_slices is not None and len(items) != case.expected_slices:
        warning.append(f"dicom_count_mismatch_expected_{case.expected_slices}_actual_{len(items)}")
    if case.expected_slices is not None and len(image_3d.shape) >= 3 and image_3d.shape[2] != case.expected_slices:
        warning.append(f"nifti_z_mismatch_expected_{case.expected_slices}_actual_{image_3d.shape[2]}")

    return {
        "patient_id": case.patient_id,
        "exam_date": exam_date,
        "selection_method": "series_uid_across_patient" if case.final_ct_series_uid else "listed_folder_no_series_uid",
        "final_ct_series_uid": case.final_ct_series_uid,
        "listed_ct_folder": str(source_dir / case.patient_id / case.final_ct_relative_path),
        "expected_slices_from_resolved_csv": case.expected_slices,
        "actual_matching_ct_dicom_files": len(items),
        "source_folder_counts": "|".join(f"{key}:{value}" for key, value in sorted(folder_counts.items())),
        "output_nifti": str(output_path),
        "nifti_shape": nifti_shape,
        "voxel_size_mm": voxel_size,
        "status": "converted",
        "warning": "|".join(warning),
        "transfer_syntax_uid": "|".join(sorted(transfer_syntaxes)),
    }


def main() -> int:
    """Run planning CT NIfTI export."""
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    resolved_csv = args.resolved_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cases = load_cases(resolved_csv, args.limit)

    if args.dry_run:
        print("Dry-run planning CT exports:")
        for case in cases:
            exam_date = date_from_relative_path(case.final_ct_relative_path)
            output_path = output_dir / f"{case.patient_id}_{exam_date}_planningCT.nii.gz"
            method = "series_uid_across_patient" if case.final_ct_series_uid else "listed_folder_no_series_uid"
            print(f"- {case.patient_id}: {method}")
            print(f"  input: {source_dir / case.patient_id / case.final_ct_relative_path}")
            print(f"  output: {output_path}")
        print(f"QC output: {output_dir / args.qc_name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / args.qc_name
    if qc_path.exists() and not args.overwrite:
        raise FileExistsError(f"QC output exists, use --overwrite to replace: {qc_path}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(export_case(source_dir, output_dir, case, args.overwrite))
    pd.DataFrame(rows).to_csv(qc_path, index=False)
    print(f"Wrote {len(rows)} planning CT NIfTI files")
    print(f"QC: {qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
