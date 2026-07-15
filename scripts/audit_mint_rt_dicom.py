#!/usr/bin/env python3
"""Audit MINT radiotherapy DICOM data without modifying source files."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom
import yaml
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID

from mint_knowledge_filter import load_folder_knowledge_filter


LOGGER = logging.getLogger("audit_mint_rt_dicom")

CT_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.2",
    "1.2.840.10008.5.1.4.1.1.2.1",
    "1.2.840.10008.5.1.4.1.1.2.2",
}
MR_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.4",
    "1.2.840.10008.5.1.4.1.1.4.1",
}
RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"


def safe_str(value: Any) -> str:
    """Return a stripped string, or empty string if missing."""
    if value is None:
        return ""
    text = str(value).strip()
    return text


def safe_float(value: Any) -> float | None:
    """Parse a float value when available."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sequence_to_text(values: Iterable[Any]) -> str:
    """Serialize a short sequence for CSV output."""
    items = [safe_str(item) for item in values if safe_str(item)]
    return "|".join(items)


def dotted_uid(value: Any) -> str:
    """Normalize UID-like values to a plain string."""
    if not value:
        return ""
    return safe_str(value)


def is_valid_dicom_dataset(ds: Dataset) -> bool:
    """Require core DICOM-identifying tags to avoid false positives from force=True."""
    sop_class_uid = safe_str(getattr(ds, "SOPClassUID", ""))
    sop_instance_uid = safe_str(getattr(ds, "SOPInstanceUID", ""))
    modality = safe_str(getattr(ds, "Modality", ""))
    study_uid = safe_str(getattr(ds, "StudyInstanceUID", ""))
    series_uid = safe_str(getattr(ds, "SeriesInstanceUID", ""))
    return bool(sop_class_uid and sop_instance_uid and (modality or study_uid or series_uid))


def get_first(ds: Dataset, names: Iterable[str]) -> str:
    """Return the first non-empty DICOM attribute among candidate names."""
    for name in names:
        value = getattr(ds, name, None)
        text = safe_str(value)
        if text:
            return text
    return ""


def get_nested_attr(item: Dataset, path: list[str]) -> Any:
    """Traverse nested DICOM attributes if present."""
    current: Any = item
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def iter_datasets(items: Any) -> Iterable[Dataset]:
    """Yield datasets from a DICOM sequence-like object."""
    if not items:
        return []
    return [entry for entry in items if isinstance(entry, Dataset)]


def classify_dicom(modality: str, sop_class_uid: str) -> str:
    """Classify a DICOM object based on Modality and SOP Class."""
    if sop_class_uid in CT_SOP_CLASSES or modality == "CT":
        return "CT"
    if sop_class_uid == RTDOSE_SOP_CLASS or modality == "RTDOSE":
        return "RTDOSE"
    if sop_class_uid == RTPLAN_SOP_CLASS or modality == "RTPLAN":
        return "RTPLAN"
    if sop_class_uid == RTSTRUCT_SOP_CLASS or modality == "RTSTRUCT":
        return "RTSTRUCT"
    if sop_class_uid in MR_SOP_CLASSES or modality == "MR":
        return "MR"
    return "other"


@dataclass
class DicomRecord:
    """Minimal per-file DICOM audit record."""

    file_path: str
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    sop_class_uid: str
    modality: str
    object_class: str
    frame_of_reference_uid: str
    study_description: str
    series_description: str
    manufacturer: str
    rows: int | None = None
    columns: int | None = None
    number_of_frames: int | None = None
    slice_thickness: float | None = None
    pixel_spacing: str = ""
    image_orientation: str = ""
    image_position: str = ""
    convolution_kernel: str = ""
    contrast_information: str = ""
    dose_units: str = ""
    dose_type: str = ""
    dose_summation_type: str = ""
    dose_grid_scaling: float | None = None
    grid_frame_offset_vector: str = ""
    referenced_rtplan_uid: str = ""
    referenced_rtstruct_uid: str = ""
    referenced_ct_series_uid: str = ""
    rtplan_label: str = ""
    rtplan_name: str = ""
    rtplan_date: str = ""
    rtplan_time: str = ""
    approval_status: str = ""
    fractions_planned: int | None = None
    prescription_dose_refs: str = ""
    prescription_target_refs: str = ""


@dataclass
class CtSeriesSummary:
    """Aggregated CT series information."""

    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    series_description: str
    study_description: str
    number_of_slices: int = 0
    slice_thicknesses: Counter[str] = field(default_factory=Counter)
    pixel_spacings: Counter[str] = field(default_factory=Counter)
    image_orientations: Counter[str] = field(default_factory=Counter)
    convolution_kernels: Counter[str] = field(default_factory=Counter)
    contrast_infos: Counter[str] = field(default_factory=Counter)
    z_positions: list[float] = field(default_factory=list)
    image_positions: list[str] = field(default_factory=list)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config if present."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def setup_logging(log_path: Path, level: str) -> None:
    """Configure console and file logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOGGER.level)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(LOGGER.level)
    LOGGER.addHandler(stream_handler)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Source DICOM root directory")
    parser.add_argument("--output-dir", type=Path, help="Output workspace directory")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Scan only and print summary")
    parser.add_argument("--max-patients", type=int, default=None, help="Limit number of PatientIDs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--knowledge-xlsx",
        type=Path,
        default=None,
        help="Optional Excel knowledge table used to keep only listed ID-Date folders",
    )
    parser.add_argument(
        "--knowledge-sheet",
        default=None,
        help="Sheet name for --knowledge-xlsx; default comes from config or RT243-3012-final",
    )
    return parser.parse_args()


def ensure_output_targets(output_dir: Path, overwrite: bool, dry_run: bool) -> dict[str, Path]:
    """Prepare output paths and enforce overwrite rules."""
    outputs = {
        "inventory": output_dir / "outputs" / "mint_dicom_inventory.csv",
        "patient_summary": output_dir / "outputs" / "mint_patient_summary.csv",
        "case_summary": output_dir / "outputs" / "mint_rt_case_summary.csv",
        "linkage_qc": output_dir / "outputs" / "mint_rt_linkage_qc.csv",
        "exclusion": output_dir / "outputs" / "mint_rt_exclusion_candidates.csv",
        "log": output_dir / "logs" / "audit_mint_rt_dicom.log",
    }
    if dry_run:
        return outputs
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for name, path in outputs.items() if name != "log" and path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist, use --overwrite: {existing_text}")
    return outputs


def list_all_files(source_dir: Path) -> list[Path]:
    """Recursively list all files under the source directory."""
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    files.sort()
    return files


def read_dicom_metadata(path: Path) -> DicomRecord | None:
    """Read a DICOM file with pydicom and extract audit metadata."""
    try:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except InvalidDicomError:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except (InvalidDicomError, FileNotFoundError, PermissionError, OSError) as exc:
        LOGGER.debug("Skipping non-DICOM or unreadable file %s: %s", path, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Unexpected error reading %s: %s", path, exc)
        return None

    if not is_valid_dicom_dataset(ds):
        return None

    sop_class_uid = dotted_uid(getattr(ds, "SOPClassUID", ""))
    sop_instance_uid = dotted_uid(getattr(ds, "SOPInstanceUID", ""))
    modality = safe_str(getattr(ds, "Modality", "")).upper()
    object_class = classify_dicom(modality, sop_class_uid)

    record = DicomRecord(
        file_path=str(path),
        patient_id=safe_str(getattr(ds, "PatientID", "")),
        study_instance_uid=dotted_uid(getattr(ds, "StudyInstanceUID", "")),
        series_instance_uid=dotted_uid(getattr(ds, "SeriesInstanceUID", "")),
        sop_instance_uid=sop_instance_uid,
        sop_class_uid=sop_class_uid,
        modality=modality,
        object_class=object_class,
        frame_of_reference_uid=dotted_uid(getattr(ds, "FrameOfReferenceUID", "")),
        study_description=safe_str(getattr(ds, "StudyDescription", "")),
        series_description=safe_str(getattr(ds, "SeriesDescription", "")),
        manufacturer=safe_str(getattr(ds, "Manufacturer", "")),
        rows=getattr(ds, "Rows", None),
        columns=getattr(ds, "Columns", None),
        number_of_frames=getattr(ds, "NumberOfFrames", None),
        slice_thickness=safe_float(getattr(ds, "SliceThickness", None)),
        pixel_spacing=sequence_to_text(getattr(ds, "PixelSpacing", []) or []),
        image_orientation=sequence_to_text(getattr(ds, "ImageOrientationPatient", []) or []),
        image_position=sequence_to_text(getattr(ds, "ImagePositionPatient", []) or []),
        convolution_kernel=safe_str(getattr(ds, "ConvolutionKernel", "")),
        contrast_information=get_first(
            ds,
            [
                "ContrastBolusAgent",
                "ContrastBolusRoute",
                "ContrastBolusIngredient",
                "ContrastBolusVolume",
            ],
        ),
    )

    if object_class == "RTDOSE":
        record.dose_units = safe_str(getattr(ds, "DoseUnits", ""))
        record.dose_type = safe_str(getattr(ds, "DoseType", ""))
        record.dose_summation_type = safe_str(getattr(ds, "DoseSummationType", ""))
        record.dose_grid_scaling = safe_float(getattr(ds, "DoseGridScaling", None))
        record.grid_frame_offset_vector = sequence_to_text(
            getattr(ds, "GridFrameOffsetVector", []) or []
        )
        record.referenced_rtplan_uid = extract_rtdose_referenced_rtplan_uid(ds)

    if object_class == "RTPLAN":
        record.rtplan_label = safe_str(getattr(ds, "RTPlanLabel", ""))
        record.rtplan_name = safe_str(getattr(ds, "RTPlanName", ""))
        record.rtplan_date = safe_str(getattr(ds, "RTPlanDate", ""))
        record.rtplan_time = safe_str(getattr(ds, "RTPlanTime", ""))
        record.approval_status = safe_str(getattr(ds, "ApprovalStatus", ""))
        record.referenced_rtstruct_uid = extract_rtplan_referenced_rtstruct_uid(ds)
        record.fractions_planned = extract_plan_fraction_count(ds)
        (
            record.prescription_dose_refs,
            record.prescription_target_refs,
        ) = extract_prescription_fields(ds)

    if object_class == "RTSTRUCT":
        record.referenced_ct_series_uid = extract_rtstruct_referenced_ct_series_uid(ds)

    return record


def extract_rtdose_referenced_rtplan_uid(ds: Dataset) -> str:
    """Extract the RTPLAN SOP Instance UID referenced by RTDOSE."""
    refs: list[str] = []
    for seq_name in ("ReferencedRTPlanSequence", "ReferencedInstanceSequence"):
        for item in iter_datasets(getattr(ds, seq_name, None)):
            uid = dotted_uid(getattr(item, "ReferencedSOPInstanceUID", ""))
            if uid:
                refs.append(uid)
    return sequence_to_text(dict.fromkeys(refs))


def extract_rtplan_referenced_rtstruct_uid(ds: Dataset) -> str:
    """Extract the RTSTRUCT SOP Instance UID referenced by RTPLAN."""
    refs: list[str] = []
    for item in iter_datasets(getattr(ds, "ReferencedStructureSetSequence", None)):
        uid = dotted_uid(getattr(item, "ReferencedSOPInstanceUID", ""))
        if uid:
            refs.append(uid)
    return sequence_to_text(dict.fromkeys(refs))


def extract_rtstruct_referenced_ct_series_uid(ds: Dataset) -> str:
    """Extract CT SeriesInstanceUID referenced by RTSTRUCT."""
    refs: list[str] = []
    for ref_frame in iter_datasets(getattr(ds, "ReferencedFrameOfReferenceSequence", None)):
        study_seq = getattr(ref_frame, "RTReferencedStudySequence", None)
        for study_item in iter_datasets(study_seq):
            series_seq = getattr(study_item, "RTReferencedSeriesSequence", None)
            for series_item in iter_datasets(series_seq):
                uid = dotted_uid(getattr(series_item, "SeriesInstanceUID", ""))
                if uid:
                    refs.append(uid)
    return sequence_to_text(dict.fromkeys(refs))


def extract_plan_fraction_count(ds: Dataset) -> int | None:
    """Extract total planned fractions when available."""
    values: list[int] = []
    for item in iter_datasets(getattr(ds, "FractionGroupSequence", None)):
        raw = getattr(item, "NumberOfFractionsPlanned", None)
        if raw not in (None, ""):
            try:
                values.append(int(raw))
            except (TypeError, ValueError):
                continue
    if values:
        return max(values)
    return None


def extract_prescription_fields(ds: Dataset) -> tuple[str, str]:
    """Extract prescription-related dose reference fields when present."""
    dose_refs: list[str] = []
    target_refs: list[str] = []
    for item in iter_datasets(getattr(ds, "DoseReferenceSequence", None)):
        ref_num = safe_str(getattr(item, "DoseReferenceNumber", ""))
        ref_desc = safe_str(getattr(item, "DoseReferenceDescription", ""))
        target_prescription = safe_str(getattr(item, "TargetPrescriptionDose", ""))
        delivery_max = safe_str(getattr(item, "DeliveryMaximumDose", ""))
        struct_type = safe_str(getattr(item, "DoseReferenceStructureType", ""))
        parts = [part for part in (ref_num, ref_desc, target_prescription, delivery_max) if part]
        if parts:
            dose_refs.append(":".join(parts))
        target_text = ":".join(part for part in (ref_num, struct_type, ref_desc) if part)
        if target_text:
            target_refs.append(target_text)
    return sequence_to_text(dose_refs), sequence_to_text(target_refs)


def build_ct_series_summaries(records: Iterable[DicomRecord]) -> dict[str, CtSeriesSummary]:
    """Aggregate per-slice CT records to per-series summaries."""
    series_map: dict[str, CtSeriesSummary] = {}
    for record in records:
        if record.object_class != "CT" or not record.series_instance_uid:
            continue
        key = record.series_instance_uid
        summary = series_map.get(key)
        if summary is None:
            summary = CtSeriesSummary(
                patient_id=record.patient_id,
                study_instance_uid=record.study_instance_uid,
                series_instance_uid=record.series_instance_uid,
                frame_of_reference_uid=record.frame_of_reference_uid,
                series_description=record.series_description,
                study_description=record.study_description,
            )
            series_map[key] = summary
        summary.number_of_slices += 1
        if record.slice_thickness is not None:
            summary.slice_thicknesses[f"{record.slice_thickness:g}"] += 1
        if record.pixel_spacing:
            summary.pixel_spacings[record.pixel_spacing] += 1
        if record.image_orientation:
            summary.image_orientations[record.image_orientation] += 1
        if record.convolution_kernel:
            summary.convolution_kernels[record.convolution_kernel] += 1
        if record.contrast_information:
            summary.contrast_infos[record.contrast_information] += 1
        if record.image_position:
            summary.image_positions.append(record.image_position)
            coords = [safe_float(item) for item in record.image_position.split("|")]
            if len(coords) == 3 and coords[2] is not None:
                summary.z_positions.append(coords[2])
    return series_map


def most_common(counter: Counter[str]) -> str:
    """Return the most common non-empty counter key."""
    return counter.most_common(1)[0][0] if counter else ""


def z_position_range(summary: CtSeriesSummary) -> str:
    """Return image position z range text."""
    if not summary.z_positions:
        return ""
    z_min = min(summary.z_positions)
    z_max = max(summary.z_positions)
    return f"{z_min:g}..{z_max:g}"


def split_refs(text: str) -> list[str]:
    """Split a pipe-delimited UID list."""
    return [item for item in text.split("|") if item]


def select_candidate_case(
    patient_id: str,
    patient_records: list[DicomRecord],
    ct_summaries: dict[str, CtSeriesSummary],
) -> dict[str, Any]:
    """Select a candidate linked RT case and report audit status."""
    plans = [record for record in patient_records if record.object_class == "RTPLAN"]
    doses = [record for record in patient_records if record.object_class == "RTDOSE"]
    structs = [record for record in patient_records if record.object_class == "RTSTRUCT"]
    cts = [record for record in patient_records if record.object_class == "CT"]

    warnings: list[str] = []
    status = "no_rt_case"

    preferred_doses = [dose for dose in doses if dose.dose_summation_type.upper() == "PLAN"]
    if len(preferred_doses) > 1:
        warnings.append("multiple_PLAN_rtdose")
    if len(doses) > 1:
        warnings.append("multiple_rtdose")

    approved_plans = [plan for plan in plans if plan.approval_status.upper() == "APPROVED"]
    if len(plans) > 1:
        warnings.append("multiple_rtplan")

    selected_dose = preferred_doses[0] if len(preferred_doses) == 1 else (preferred_doses[0] if preferred_doses else (doses[0] if len(doses) == 1 else None))
    selected_plan = approved_plans[0] if len(approved_plans) == 1 else (approved_plans[0] if approved_plans else (plans[0] if len(plans) == 1 else None))

    if selected_dose is None and doses:
        status = "ambiguous"
        warnings.append("dose_selection_ambiguous")
    if selected_plan is None and plans:
        status = "ambiguous"
        warnings.append("plan_selection_ambiguous")

    selected_struct: DicomRecord | None = None
    selected_ct_series_uid = ""
    selected_frame_match = ""

    if selected_plan and selected_plan.referenced_rtstruct_uid:
        ref_structs = {
            record.sop_instance_uid: record for record in structs if record.sop_instance_uid
        }
        plan_struct_uids = split_refs(selected_plan.referenced_rtstruct_uid)
        matched_structs = [ref_structs[uid] for uid in plan_struct_uids if uid in ref_structs]
        if len(matched_structs) == 1:
            selected_struct = matched_structs[0]
        elif len(matched_structs) > 1:
            status = "ambiguous"
            warnings.append("multiple_rtstruct_for_plan")
    elif len(structs) == 1:
        selected_struct = structs[0]

    if selected_struct and selected_struct.referenced_ct_series_uid:
        ct_ref_uids = split_refs(selected_struct.referenced_ct_series_uid)
        matched_cts = [uid for uid in ct_ref_uids if uid in ct_summaries]
        if len(matched_cts) == 1:
            selected_ct_series_uid = matched_cts[0]
        elif len(matched_cts) > 1:
            status = "ambiguous"
            warnings.append("multiple_ct_series_in_rtstruct")
    else:
        if structs and not selected_struct:
            warnings.append("rtstruct_not_linked_to_selected_plan")

    if selected_dose and selected_plan:
        dose_plan_refs = split_refs(selected_dose.referenced_rtplan_uid)
        if dose_plan_refs and selected_plan.sop_instance_uid not in dose_plan_refs:
            status = "ambiguous"
            warnings.append("dose_plan_reference_mismatch")

    if selected_ct_series_uid:
        ct_frame = ct_summaries[selected_ct_series_uid].frame_of_reference_uid
        frames = {
            "ct": ct_frame,
            "rtstruct": selected_struct.frame_of_reference_uid if selected_struct else "",
            "rtplan": selected_plan.frame_of_reference_uid if selected_plan else "",
            "rtdose": selected_dose.frame_of_reference_uid if selected_dose else "",
        }
        non_empty_frames = {value for value in frames.values() if value}
        selected_frame_match = "yes" if len(non_empty_frames) <= 1 else "no"
        if selected_frame_match == "no":
            status = "ambiguous"
            warnings.append("frame_of_reference_mismatch")

    if any(marker in warnings for marker in ("multiple_rtplan", "multiple_rtdose")):
        warnings.append("possible_reirradiation_or_boost")

    has_complete = bool(cts and plans and doses and structs and selected_ct_series_uid and selected_struct and selected_plan and selected_dose)
    if has_complete and status != "ambiguous":
        status = "complete"
    elif status != "ambiguous" and (plans or doses or structs):
        status = "incomplete"

    return {
        "patient_id": patient_id,
        "n_ct_series": len({record.series_instance_uid for record in cts if record.series_instance_uid}),
        "n_rtplan": len(plans),
        "n_rtdose": len(doses),
        "n_rtstruct": len(structs),
        "n_mr_series": len({record.series_instance_uid for record in patient_records if record.object_class == "MR" and record.series_instance_uid}),
        "has_complete_rt_set": has_complete,
        "candidate_planning_ct_series_uid": selected_ct_series_uid,
        "candidate_plan_uid": selected_plan.sop_instance_uid if selected_plan else "",
        "candidate_dose_uid": selected_dose.sop_instance_uid if selected_dose else "",
        "candidate_structure_uid": selected_struct.sop_instance_uid if selected_struct else "",
        "frame_of_reference_match": selected_frame_match,
        "dose_summation_type": selected_dose.dose_summation_type if selected_dose else "",
        "dose_units": selected_dose.dose_units if selected_dose else "",
        "n_fractions": selected_plan.fractions_planned if selected_plan else None,
        "audit_status": status,
        "audit_warning": sequence_to_text(dict.fromkeys(warnings)),
        "selected_plan_record": selected_plan,
        "selected_dose_record": selected_dose,
        "selected_struct_record": selected_struct,
    }


def build_inventory_rows(records: list[DicomRecord], ct_summaries: dict[str, CtSeriesSummary]) -> list[dict[str, Any]]:
    """Build file-level inventory output."""
    rows: list[dict[str, Any]] = []
    for record in records:
        ct_summary = ct_summaries.get(record.series_instance_uid)
        rows.append(
            {
                "file_path": record.file_path,
                "patient_id": record.patient_id,
                "study_instance_uid": record.study_instance_uid,
                "series_instance_uid": record.series_instance_uid,
                "sop_instance_uid": record.sop_instance_uid,
                "sop_class_uid": record.sop_class_uid,
                "modality": record.modality,
                "object_class": record.object_class,
                "frame_of_reference_uid": record.frame_of_reference_uid,
                "study_description": record.study_description,
                "series_description": record.series_description,
                "rows": record.rows,
                "columns": record.columns,
                "number_of_frames": record.number_of_frames,
                "slice_thickness": record.slice_thickness,
                "pixel_spacing": record.pixel_spacing,
                "image_orientation": record.image_orientation,
                "image_position": record.image_position,
                "convolution_kernel": record.convolution_kernel,
                "contrast_information": record.contrast_information,
                "dose_units": record.dose_units,
                "dose_type": record.dose_type,
                "dose_summation_type": record.dose_summation_type,
                "dose_grid_scaling": record.dose_grid_scaling,
                "grid_frame_offset_vector": record.grid_frame_offset_vector,
                "referenced_rtplan_uid": record.referenced_rtplan_uid,
                "rtplan_label": record.rtplan_label,
                "rtplan_name": record.rtplan_name,
                "rtplan_date": record.rtplan_date,
                "rtplan_time": record.rtplan_time,
                "approval_status": record.approval_status,
                "referenced_rtstruct_uid": record.referenced_rtstruct_uid,
                "fractions_planned": record.fractions_planned,
                "prescription_dose_refs": record.prescription_dose_refs,
                "prescription_target_refs": record.prescription_target_refs,
                "referenced_ct_series_uid": record.referenced_ct_series_uid,
                "ct_number_of_slices": ct_summary.number_of_slices if ct_summary else None,
                "ct_image_position_range": z_position_range(ct_summary) if ct_summary else "",
            }
        )
    return rows


def build_case_summary_rows(
    patient_summaries: list[dict[str, Any]],
    ct_summaries: dict[str, CtSeriesSummary],
) -> list[dict[str, Any]]:
    """Build selected RT case summary rows."""
    rows: list[dict[str, Any]] = []
    for summary in patient_summaries:
        plan = summary.pop("selected_plan_record")
        dose = summary.pop("selected_dose_record")
        struct = summary.pop("selected_struct_record")
        ct_uid = summary["candidate_planning_ct_series_uid"]
        ct_summary = ct_summaries.get(ct_uid)
        rows.append(
            {
                **summary,
                "candidate_plan_label": plan.rtplan_label if plan else "",
                "candidate_plan_name": plan.rtplan_name if plan else "",
                "candidate_plan_date": plan.rtplan_date if plan else "",
                "candidate_plan_approval_status": plan.approval_status if plan else "",
                "candidate_rtstruct_ref_uid": plan.referenced_rtstruct_uid if plan else "",
                "candidate_dose_type": dose.dose_type if dose else "",
                "candidate_dose_grid_scaling": dose.dose_grid_scaling if dose else None,
                "candidate_dose_referenced_plan_uid": dose.referenced_rtplan_uid if dose else "",
                "candidate_structure_referenced_ct_series_uid": struct.referenced_ct_series_uid if struct else "",
                "ct_number_of_slices": ct_summary.number_of_slices if ct_summary else None,
                "ct_slice_thickness": most_common(ct_summary.slice_thicknesses) if ct_summary else "",
                "ct_pixel_spacing": most_common(ct_summary.pixel_spacings) if ct_summary else "",
                "ct_image_orientation": most_common(ct_summary.image_orientations) if ct_summary else "",
                "ct_image_position_range": z_position_range(ct_summary) if ct_summary else "",
                "ct_convolution_kernel": most_common(ct_summary.convolution_kernels) if ct_summary else "",
                "ct_contrast_information": most_common(ct_summary.contrast_infos) if ct_summary else "",
                "ct_series_description": ct_summary.series_description if ct_summary else "",
                "ct_study_description": ct_summary.study_description if ct_summary else "",
            }
        )
    return rows


def build_linkage_qc_rows(
    records: list[DicomRecord],
    ct_summaries: dict[str, CtSeriesSummary],
) -> list[dict[str, Any]]:
    """Build linkage QC rows for RT objects."""
    struct_map = {record.sop_instance_uid: record for record in records if record.object_class == "RTSTRUCT"}
    plan_map = {record.sop_instance_uid: record for record in records if record.object_class == "RTPLAN"}
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.object_class not in {"RTDOSE", "RTPLAN", "RTSTRUCT"}:
            continue
        linked_struct_uid = ""
        linked_ct_uid = ""
        frame_match = ""
        if record.object_class == "RTDOSE":
            plan_uids = split_refs(record.referenced_rtplan_uid)
            linked_plan = plan_map.get(plan_uids[0]) if len(plan_uids) == 1 else None
            linked_struct_uid = linked_plan.referenced_rtstruct_uid if linked_plan else ""
            if linked_plan:
                struct_uids = split_refs(linked_plan.referenced_rtstruct_uid)
                linked_struct = struct_map.get(struct_uids[0]) if len(struct_uids) == 1 else None
                if linked_struct:
                    linked_ct_uid = linked_struct.referenced_ct_series_uid
                    frames = {
                        value
                        for value in (
                            record.frame_of_reference_uid,
                            linked_plan.frame_of_reference_uid,
                            linked_struct.frame_of_reference_uid,
                            ct_summaries.get(linked_ct_uid, CtSeriesSummary("", "", "", "", "", "")).frame_of_reference_uid
                            if linked_ct_uid
                            else "",
                        )
                        if value
                    }
                    frame_match = "yes" if len(frames) <= 1 else "no"
        elif record.object_class == "RTPLAN":
            linked_struct_uid = record.referenced_rtstruct_uid
            struct_uids = split_refs(record.referenced_rtstruct_uid)
            linked_struct = struct_map.get(struct_uids[0]) if len(struct_uids) == 1 else None
            if linked_struct:
                linked_ct_uid = linked_struct.referenced_ct_series_uid
                frames = {
                    value
                    for value in (
                        record.frame_of_reference_uid,
                        linked_struct.frame_of_reference_uid,
                        ct_summaries.get(linked_ct_uid, CtSeriesSummary("", "", "", "", "", "")).frame_of_reference_uid
                        if linked_ct_uid
                        else "",
                    )
                    if value
                }
                frame_match = "yes" if len(frames) <= 1 else "no"
        else:
            linked_ct_uid = record.referenced_ct_series_uid
            frames = {
                value
                for value in (
                    record.frame_of_reference_uid,
                    ct_summaries.get(linked_ct_uid, CtSeriesSummary("", "", "", "", "", "")).frame_of_reference_uid
                    if linked_ct_uid
                    else "",
                )
                if value
            }
            frame_match = "yes" if len(frames) <= 1 else "no"
        rows.append(
            {
                "patient_id": record.patient_id,
                "object_class": record.object_class,
                "sop_instance_uid": record.sop_instance_uid,
                "series_instance_uid": record.series_instance_uid,
                "frame_of_reference_uid": record.frame_of_reference_uid,
                "referenced_rtplan_uid": record.referenced_rtplan_uid,
                "referenced_rtstruct_uid": linked_struct_uid if record.object_class != "RTPLAN" else record.referenced_rtstruct_uid,
                "referenced_ct_series_uid": linked_ct_uid,
                "frame_of_reference_match": frame_match,
            }
        )
    return rows


def build_exclusion_rows(patient_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build exclusion or follow-up candidate rows."""
    rows: list[dict[str, Any]] = []
    for summary in patient_summaries:
        if summary["has_complete_rt_set"] and summary["audit_status"] == "complete":
            continue
        rows.append(
            {
                "patient_id": summary["patient_id"],
                "audit_status": summary["audit_status"],
                "audit_warning": summary["audit_warning"],
                "n_ct_series": summary["n_ct_series"],
                "n_rtplan": summary["n_rtplan"],
                "n_rtdose": summary["n_rtdose"],
                "n_rtstruct": summary["n_rtstruct"],
                "candidate_plan_uid": summary["candidate_plan_uid"],
                "candidate_dose_uid": summary["candidate_dose_uid"],
                "candidate_structure_uid": summary["candidate_structure_uid"],
                "candidate_planning_ct_series_uid": summary["candidate_planning_ct_series_uid"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def print_summary(stats: dict[str, int]) -> None:
    """Print final audit counts."""
    ordered_keys = [
        "total patients",
        "patients with CT",
        "patients with RTPLAN",
        "patients with PLAN RTDOSE",
        "patients with RTSTRUCT",
        "complete linked RT cases",
        "ambiguous cases",
        "FrameOfReference mismatches",
        "possible reirradiation cases",
    ]
    for key in ordered_keys:
        print(f"{key}: {stats.get(key, 0)}")


def main() -> int:
    """Run the audit pipeline."""
    args = parse_args()
    config = load_config(args.config)

    source_dir = (args.source_dir or Path(config.get("source_dir", ""))).expanduser().resolve()
    output_dir = (args.output_dir or Path(config.get("output_dir", "."))).expanduser().resolve()
    outputs = ensure_output_targets(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)
    setup_logging(outputs["log"], safe_str(config.get("log_level", "INFO")) or "INFO")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    knowledge_xlsx = args.knowledge_xlsx or (
        Path(config["knowledge_xlsx"]) if config.get("knowledge_xlsx") else None
    )
    knowledge_sheet = args.knowledge_sheet or safe_str(
        config.get("knowledge_sheet", "RT243-3012-final")
    )
    folder_filter = load_folder_knowledge_filter(knowledge_xlsx, knowledge_sheet)
    if folder_filter is not None:
        LOGGER.info(
            "Loaded folder knowledge filter: %s sheet=%s rows=%d allowed_pairs=%d",
            folder_filter.source_path,
            folder_filter.sheet_name,
            folder_filter.n_rows,
            len(folder_filter.pairs),
        )

    LOGGER.info("Scanning source directory: %s", source_dir)
    all_files = list_all_files(source_dir)
    LOGGER.info("Found %d total files under source directory", len(all_files))
    if folder_filter is not None:
        before_filter = len(all_files)
        all_files = [
            path
            for path in all_files
            if folder_filter.matches_relative_path(path.relative_to(source_dir))
        ]
        LOGGER.info(
            "Knowledge filter kept %d/%d files in listed ID-Date folders",
            len(all_files),
            before_filter,
        )

    records: list[DicomRecord] = []
    allowed_patients: set[str] | None = None
    if args.max_patients is not None:
        patient_order: list[str] = []
        for path in all_files:
            patient_id = path.relative_to(source_dir).parts[0] if path.relative_to(source_dir).parts else ""
            if patient_id and patient_id not in patient_order:
                patient_order.append(patient_id)
            if len(patient_order) >= args.max_patients:
                break
        allowed_patients = set(patient_order)
        LOGGER.info("Limiting scan to first %d patient directories", len(allowed_patients))

    non_dicom_count = 0
    for path in all_files:
        rel_parts = path.relative_to(source_dir).parts
        top_level = rel_parts[0] if rel_parts else ""
        if allowed_patients is not None and top_level not in allowed_patients:
            continue
        record = read_dicom_metadata(path)
        if record is None:
            non_dicom_count += 1
            continue
        records.append(record)

    LOGGER.info("Detected %d DICOM files; skipped %d non-DICOM/unreadable files", len(records), non_dicom_count)

    ct_summaries = build_ct_series_summaries(records)
    records_by_patient: dict[str, list[DicomRecord]] = defaultdict(list)
    for record in records:
        records_by_patient[record.patient_id or "UNKNOWN"].append(record)

    patient_summaries: list[dict[str, Any]] = []
    for patient_id, patient_records in sorted(records_by_patient.items()):
        patient_summaries.append(select_candidate_case(patient_id, patient_records, ct_summaries))

    inventory_rows = build_inventory_rows(records, ct_summaries)
    case_rows = build_case_summary_rows(patient_summaries, ct_summaries)
    linkage_rows = build_linkage_qc_rows(records, ct_summaries)
    exclusion_rows = build_exclusion_rows(case_rows)

    patient_summary_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"candidate_plan_label", "candidate_plan_name", "candidate_plan_date", "candidate_plan_approval_status", "candidate_rtstruct_ref_uid", "candidate_dose_type", "candidate_dose_grid_scaling", "candidate_dose_referenced_plan_uid", "candidate_structure_referenced_ct_series_uid", "ct_number_of_slices", "ct_slice_thickness", "ct_pixel_spacing", "ct_image_orientation", "ct_image_position_range", "ct_convolution_kernel", "ct_contrast_information", "ct_series_description", "ct_study_description"}
        }
        for row in case_rows
    ]

    stats = {
        "total patients": len(patient_summaries),
        "patients with CT": sum(1 for row in patient_summary_rows if row["n_ct_series"] > 0),
        "patients with RTPLAN": sum(1 for row in patient_summary_rows if row["n_rtplan"] > 0),
        "patients with PLAN RTDOSE": sum(1 for row in patient_summary_rows if row["dose_summation_type"].upper() == "PLAN"),
        "patients with RTSTRUCT": sum(1 for row in patient_summary_rows if row["n_rtstruct"] > 0),
        "complete linked RT cases": sum(1 for row in patient_summary_rows if row["audit_status"] == "complete"),
        "ambiguous cases": sum(1 for row in patient_summary_rows if row["audit_status"] == "ambiguous"),
        "FrameOfReference mismatches": sum(1 for row in patient_summary_rows if "frame_of_reference_mismatch" in row["audit_warning"]),
        "possible reirradiation cases": sum(1 for row in patient_summary_rows if "possible_reirradiation_or_boost" in row["audit_warning"]),
    }

    if args.dry_run:
        print("Dry-run output targets:")
        for key, path in outputs.items():
            if key == "log":
                continue
            print(f"- {key}: {path}")
        print_summary(stats)
        return 0

    write_csv(outputs["inventory"], inventory_rows)
    write_csv(outputs["patient_summary"], patient_summary_rows)
    write_csv(outputs["case_summary"], case_rows)
    write_csv(outputs["linkage_qc"], linkage_rows)
    write_csv(outputs["exclusion"], exclusion_rows)

    LOGGER.info("Wrote inventory to %s", outputs["inventory"])
    LOGGER.info("Wrote patient summary to %s", outputs["patient_summary"])
    LOGGER.info("Wrote RT case summary to %s", outputs["case_summary"])
    LOGGER.info("Wrote linkage QC to %s", outputs["linkage_qc"])
    LOGGER.info("Wrote exclusion candidates to %s", outputs["exclusion"])

    print_summary(stats)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI safety
        LOGGER.exception("Audit failed: %s", exc)
        raise
