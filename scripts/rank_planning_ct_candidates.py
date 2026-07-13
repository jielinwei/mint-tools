#!/usr/bin/env python3
"""Rank candidate planning CT series for MINT RT cases."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError


LOGGER = logging.getLogger("rank_planning_ct_candidates")

CT_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.2",
    "1.2.840.10008.5.1.4.1.1.2.1",
    "1.2.840.10008.5.1.4.1.1.2.2",
}
RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"
POSITIVE_KEYWORDS = ("RT", "TRA", "LUNG", "CT", "THORAX", "PLAN", "PL", "BPL", "MAMMA", "BOOST", "BST")
NEGATIVE_KEYWORDS = ("LOCALIZER", "SCOUT", "TOPOGRAM", "COR", "SAG", "MIP", "REFORMAT")


def safe_str(value: Any) -> str:
    """Return a stripped string, or empty string if missing."""
    return "" if value is None else str(value).strip()


def safe_float(value: Any) -> float | None:
    """Parse float-like values."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """Parse int-like values."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def seq_text(values: Iterable[Any]) -> str:
    """Serialize a short sequence to pipe-delimited text."""
    return "|".join(safe_str(item) for item in values if safe_str(item))


def split_refs(text: str) -> list[str]:
    """Split pipe-delimited UID text."""
    return [item for item in safe_str(text).split("|") if item]


def classify_dicom(modality: str, sop_class_uid: str) -> str:
    """Classify a DICOM instance."""
    modality = modality.upper()
    if sop_class_uid in CT_SOP_CLASSES or modality == "CT":
        return "CT"
    if sop_class_uid == RTDOSE_SOP_CLASS or modality == "RTDOSE":
        return "RTDOSE"
    if sop_class_uid == RTPLAN_SOP_CLASS or modality == "RTPLAN":
        return "RTPLAN"
    if sop_class_uid == RTSTRUCT_SOP_CLASS or modality == "RTSTRUCT":
        return "RTSTRUCT"
    if modality == "MR":
        return "MR"
    return "other"


def is_valid_dicom_dataset(ds: Dataset) -> bool:
    """Require core DICOM-identifying tags to avoid false positives from force=True."""
    sop_class_uid = safe_str(getattr(ds, "SOPClassUID", ""))
    sop_instance_uid = safe_str(getattr(ds, "SOPInstanceUID", ""))
    modality = safe_str(getattr(ds, "Modality", ""))
    study_uid = safe_str(getattr(ds, "StudyInstanceUID", ""))
    series_uid = safe_str(getattr(ds, "SeriesInstanceUID", ""))
    return bool(sop_class_uid and sop_instance_uid and (modality or study_uid or series_uid))


@dataclass
class DicomFileRecord:
    """Per-file metadata record."""

    file_path: Path
    relative_path: Path
    patient_id: str
    exam_folder_name: str
    exam_date: str
    study_date: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    sop_class_uid: str
    modality: str
    object_class: str
    frame_of_reference_uid: str
    series_description: str
    protocol_name: str
    study_description: str
    convolution_kernel: str
    slice_thickness: float | None
    spacing_between_slices: float | None
    pixel_spacing: str
    image_orientation: str
    image_position: tuple[float, float, float] | None
    rows: int | None
    columns: int | None
    number_of_frames: int | None
    grid_frame_offset_vector: list[float] = field(default_factory=list)
    dose_units: str = ""
    dose_type: str = ""
    dose_summation_type: str = ""
    rtplan_label: str = ""
    rtplan_name: str = ""
    rtplan_date: str = ""
    approval_status: str = ""
    fractions_planned: int | None = None
    referenced_rtplan_uid: str = ""
    referenced_rtstruct_uid: str = ""
    referenced_ct_series_uid: str = ""


@dataclass
class CtSeriesInfo:
    """Aggregated CT series metadata."""

    patient_id: str
    exam_folder_name: str
    exam_date: str
    study_date: str
    relative_path: str
    path_depth: int
    series_folder_name: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    series_description: str
    protocol_name: str
    study_description: str
    number_of_ct_slices: int = 0
    slice_thicknesses: Counter[str] = field(default_factory=Counter)
    spacing_between_slices: Counter[str] = field(default_factory=Counter)
    pixel_spacings: Counter[str] = field(default_factory=Counter)
    image_orientations: Counter[str] = field(default_factory=Counter)
    convolution_kernels: Counter[str] = field(default_factory=Counter)
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    rows: Counter[int] = field(default_factory=Counter)
    columns: Counter[int] = field(default_factory=Counter)


@dataclass
class DoseInfo:
    """RTDOSE aggregated metadata."""

    patient_id: str
    study_date: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    frame_of_reference_uid: str
    referenced_rtplan_uid: str
    dose_summation_type: str
    number_of_frames: int | None
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    image_position: tuple[float, float, float] | None
    z_offsets: list[float]


@dataclass
class PlanInfo:
    """RTPLAN metadata."""

    patient_id: str
    study_date: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    frame_of_reference_uid: str
    rtplan_label: str
    rtplan_name: str
    rtplan_date: str
    approval_status: str
    fractions_planned: int | None
    referenced_rtstruct_uid: str


@dataclass
class StructInfo:
    """RTSTRUCT metadata."""

    patient_id: str
    study_date: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    frame_of_reference_uid: str
    referenced_ct_series_uid: str


@dataclass
class RtCourse:
    """Grouped RT course metadata."""

    patient_id: str
    rt_course_id: str
    plan: PlanInfo | None
    dose: DoseInfo | None
    struct: StructInfo | None
    course_date: str
    frame_of_reference_uid: str
    course_type: str
    warning: str


def parse_date(text: str) -> datetime | None:
    """Parse a DICOM or folder date."""
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def find_exam_folder_name(relative_path: Path) -> tuple[str, str]:
    """Infer the examination date folder from a path relative to the source root."""
    parts = relative_path.parts
    exam_folder = parts[1] if len(parts) > 1 else ""
    match = re.match(r"^(\d{8})", exam_folder)
    exam_date = match.group(1) if match else ""
    return exam_folder, exam_date


def text_contains_keyword(text: str, keyword: str) -> bool:
    """Keyword match with path-safe normalization."""
    text = safe_str(text).upper()
    if not text:
        return False
    return keyword in text


def keyword_flags(texts: Iterable[str]) -> dict[str, bool]:
    """Compute keyword presence across a list of texts."""
    combined = " / ".join(safe_str(text).upper() for text in texts if safe_str(text))
    return {keyword: text_contains_keyword(combined, keyword) for keyword in POSITIVE_KEYWORDS}


def negative_match(texts: Iterable[str]) -> list[str]:
    """Return negative keywords found."""
    combined = " / ".join(safe_str(text).upper() for text in texts if safe_str(text))
    return [keyword for keyword in NEGATIVE_KEYWORDS if keyword in combined]


def read_dicom_file(path: Path, source_dir: Path) -> DicomFileRecord | None:
    """Read DICOM metadata without pixel data."""
    try:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except InvalidDicomError:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except (InvalidDicomError, FileNotFoundError, PermissionError, OSError):
        return None
    if not is_valid_dicom_dataset(ds):
        return None

    relative_path = path.relative_to(source_dir)
    patient_id = safe_str(getattr(ds, "PatientID", "")) or (relative_path.parts[0] if relative_path.parts else "")
    exam_folder_name, exam_date = find_exam_folder_name(relative_path)
    sop_class_uid = safe_str(getattr(ds, "SOPClassUID", ""))
    modality = safe_str(getattr(ds, "Modality", ""))
    object_class = classify_dicom(modality, sop_class_uid)
    image_position_values = getattr(ds, "ImagePositionPatient", None)
    image_position: tuple[float, float, float] | None = None
    if image_position_values and len(image_position_values) >= 3:
        coords = [safe_float(value) for value in image_position_values[:3]]
        if all(value is not None for value in coords):
            image_position = (coords[0] or 0.0, coords[1] or 0.0, coords[2] or 0.0)

    record = DicomFileRecord(
        file_path=path,
        relative_path=relative_path,
        patient_id=patient_id,
        exam_folder_name=exam_folder_name,
        exam_date=exam_date,
        study_date=safe_str(getattr(ds, "StudyDate", "")),
        study_instance_uid=safe_str(getattr(ds, "StudyInstanceUID", "")),
        series_instance_uid=safe_str(getattr(ds, "SeriesInstanceUID", "")),
        sop_instance_uid=safe_str(getattr(ds, "SOPInstanceUID", "")),
        sop_class_uid=sop_class_uid,
        modality=modality.upper(),
        object_class=object_class,
        frame_of_reference_uid=safe_str(getattr(ds, "FrameOfReferenceUID", "")),
        series_description=safe_str(getattr(ds, "SeriesDescription", "")),
        protocol_name=safe_str(getattr(ds, "ProtocolName", "")),
        study_description=safe_str(getattr(ds, "StudyDescription", "")),
        convolution_kernel=safe_str(getattr(ds, "ConvolutionKernel", "")),
        slice_thickness=safe_float(getattr(ds, "SliceThickness", None)),
        spacing_between_slices=safe_float(getattr(ds, "SpacingBetweenSlices", None)),
        pixel_spacing=seq_text(getattr(ds, "PixelSpacing", []) or []),
        image_orientation=seq_text(getattr(ds, "ImageOrientationPatient", []) or []),
        image_position=image_position,
        rows=safe_int(getattr(ds, "Rows", None)),
        columns=safe_int(getattr(ds, "Columns", None)),
        number_of_frames=safe_int(getattr(ds, "NumberOfFrames", None)),
    )

    if object_class == "RTDOSE":
        record.grid_frame_offset_vector = [
            value
            for value in (safe_float(item) for item in getattr(ds, "GridFrameOffsetVector", []) or [])
            if value is not None
        ]
        record.dose_units = safe_str(getattr(ds, "DoseUnits", ""))
        record.dose_type = safe_str(getattr(ds, "DoseType", ""))
        record.dose_summation_type = safe_str(getattr(ds, "DoseSummationType", ""))
        refs = []
        for item in getattr(ds, "ReferencedRTPlanSequence", []) or []:
            uid = safe_str(getattr(item, "ReferencedSOPInstanceUID", ""))
            if uid:
                refs.append(uid)
        record.referenced_rtplan_uid = seq_text(dict.fromkeys(refs))

    if object_class == "RTPLAN":
        record.rtplan_label = safe_str(getattr(ds, "RTPlanLabel", ""))
        record.rtplan_name = safe_str(getattr(ds, "RTPlanName", ""))
        record.rtplan_date = safe_str(getattr(ds, "RTPlanDate", "")) or record.study_date
        record.approval_status = safe_str(getattr(ds, "ApprovalStatus", ""))
        refs = []
        for item in getattr(ds, "ReferencedStructureSetSequence", []) or []:
            uid = safe_str(getattr(item, "ReferencedSOPInstanceUID", ""))
            if uid:
                refs.append(uid)
        record.referenced_rtstruct_uid = seq_text(dict.fromkeys(refs))
        fractions = []
        for item in getattr(ds, "FractionGroupSequence", []) or []:
            value = safe_int(getattr(item, "NumberOfFractionsPlanned", None))
            if value is not None:
                fractions.append(value)
        record.fractions_planned = max(fractions) if fractions else None

    if object_class == "RTSTRUCT":
        refs = []
        for ref_frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []) or []:
            for ref_study in getattr(ref_frame, "RTReferencedStudySequence", []) or []:
                for ref_series in getattr(ref_study, "RTReferencedSeriesSequence", []) or []:
                    uid = safe_str(getattr(ref_series, "SeriesInstanceUID", ""))
                    if uid:
                        refs.append(uid)
        record.referenced_ct_series_uid = seq_text(dict.fromkeys(refs))

    return record


def scan_source(source_dir: Path) -> list[DicomFileRecord]:
    """Recursively scan source directory for DICOM files."""
    records: list[DicomFileRecord] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        record = read_dicom_file(path, source_dir)
        if record is not None:
            records.append(record)
    return records


def most_common(counter: Counter[Any]) -> Any:
    """Return most common counter key."""
    return counter.most_common(1)[0][0] if counter else ""


def build_ct_series(records: list[DicomFileRecord]) -> dict[str, CtSeriesInfo]:
    """Aggregate CT files into series-level metadata."""
    ct_series: dict[str, CtSeriesInfo] = {}
    for record in records:
        if record.object_class != "CT" or not record.series_instance_uid:
            continue
        info = ct_series.get(record.series_instance_uid)
        if info is None:
            relative_path = record.relative_path.parent
            info = CtSeriesInfo(
                patient_id=record.patient_id,
                exam_folder_name=record.exam_folder_name,
                exam_date=record.exam_date,
                study_date=record.study_date,
                relative_path=str(relative_path.relative_to(relative_path.parts[0])) if len(relative_path.parts) > 1 else "",
                path_depth=len(relative_path.parts),
                series_folder_name=relative_path.name,
                study_instance_uid=record.study_instance_uid,
                series_instance_uid=record.series_instance_uid,
                frame_of_reference_uid=record.frame_of_reference_uid,
                series_description=record.series_description,
                protocol_name=record.protocol_name,
                study_description=record.study_description,
            )
            ct_series[record.series_instance_uid] = info
        info.number_of_ct_slices += 1
        if record.slice_thickness is not None:
            info.slice_thicknesses[f"{record.slice_thickness:g}"] += 1
        if record.spacing_between_slices is not None:
            info.spacing_between_slices[f"{record.spacing_between_slices:g}"] += 1
        if record.pixel_spacing:
            info.pixel_spacings[record.pixel_spacing] += 1
        if record.image_orientation:
            info.image_orientations[record.image_orientation] += 1
        if record.convolution_kernel:
            info.convolution_kernels[record.convolution_kernel] += 1
        if record.image_position:
            info.positions.append(record.image_position)
        if record.rows is not None:
            info.rows[record.rows] += 1
        if record.columns is not None:
            info.columns[record.columns] += 1
    return ct_series


def build_rt_maps(records: list[DicomFileRecord]) -> tuple[dict[str, PlanInfo], dict[str, StructInfo], dict[str, DoseInfo]]:
    """Build RTPLAN/RTSTRUCT/RTDOSE maps keyed by SOPInstanceUID."""
    plans: dict[str, PlanInfo] = {}
    structs: dict[str, StructInfo] = {}
    doses: dict[str, DoseInfo] = {}
    for record in records:
        if record.object_class == "RTPLAN":
            plans[record.sop_instance_uid] = PlanInfo(
                patient_id=record.patient_id,
                study_date=record.study_date,
                study_instance_uid=record.study_instance_uid,
                series_instance_uid=record.series_instance_uid,
                sop_instance_uid=record.sop_instance_uid,
                frame_of_reference_uid=record.frame_of_reference_uid,
                rtplan_label=record.rtplan_label,
                rtplan_name=record.rtplan_name,
                rtplan_date=record.rtplan_date,
                approval_status=record.approval_status,
                fractions_planned=record.fractions_planned,
                referenced_rtstruct_uid=record.referenced_rtstruct_uid,
            )
        elif record.object_class == "RTSTRUCT":
            structs[record.sop_instance_uid] = StructInfo(
                patient_id=record.patient_id,
                study_date=record.study_date,
                study_instance_uid=record.study_instance_uid,
                series_instance_uid=record.series_instance_uid,
                sop_instance_uid=record.sop_instance_uid,
                frame_of_reference_uid=record.frame_of_reference_uid,
                referenced_ct_series_uid=record.referenced_ct_series_uid,
            )
        elif record.object_class == "RTDOSE":
            pixel_spacing_values = [safe_float(item) for item in record.pixel_spacing.split("|") if item]
            doses[record.sop_instance_uid] = DoseInfo(
                patient_id=record.patient_id,
                study_date=record.study_date,
                study_instance_uid=record.study_instance_uid,
                series_instance_uid=record.series_instance_uid,
                sop_instance_uid=record.sop_instance_uid,
                frame_of_reference_uid=record.frame_of_reference_uid,
                referenced_rtplan_uid=record.referenced_rtplan_uid,
                dose_summation_type=record.dose_summation_type,
                number_of_frames=record.number_of_frames,
                rows=record.rows,
                columns=record.columns,
                pixel_spacing=(pixel_spacing_values[0], pixel_spacing_values[1]) if len(pixel_spacing_values) >= 2 and None not in pixel_spacing_values[:2] else None,
                image_position=record.image_position,
                z_offsets=record.grid_frame_offset_vector,
            )
    return plans, structs, doses


def detect_course_type(plan: PlanInfo | None, warning_parts: list[str]) -> str:
    """Infer course type heuristically."""
    label_text = " ".join(
        safe_str(text).upper()
        for text in (
            plan.rtplan_label if plan else "",
            plan.rtplan_name if plan else "",
        )
    )
    if "BOOST" in label_text or "BST" in label_text:
        return "boost"
    if "ADAPT" in label_text or "REPLAN" in label_text:
        return "adaptive"
    if "REIRR" in label_text:
        return "reirradiation"
    if "multiple_rtplan" in warning_parts:
        return "possible_reirradiation"
    return "initial_or_unspecified"


def group_rt_courses(records: list[DicomFileRecord]) -> tuple[list[RtCourse], dict[str, int]]:
    """Group RT objects into distinct RT courses."""
    plans, structs, doses = build_rt_maps(records)
    records_by_patient: dict[str, list[DicomFileRecord]] = defaultdict(list)
    for record in records:
        records_by_patient[record.patient_id].append(record)

    courses: list[RtCourse] = []
    patient_course_counts: dict[str, int] = {}

    for patient_id, patient_records in sorted(records_by_patient.items()):
        patient_plans = [plans[key] for key in plans if plans[key].patient_id == patient_id]
        patient_doses = [doses[key] for key in doses if doses[key].patient_id == patient_id]
        if not patient_plans and not patient_doses:
            continue

        plan_to_doses: dict[str, list[DoseInfo]] = defaultdict(list)
        for dose in patient_doses:
            for plan_uid in split_refs(dose.referenced_rtplan_uid):
                plan_to_doses[plan_uid].append(dose)

        plan_order = sorted(
            patient_plans,
            key=lambda plan: (
                parse_date(plan.rtplan_date or plan.study_date) or datetime.max,
                plan.sop_instance_uid,
            ),
        )
        for index, plan in enumerate(plan_order, start=1):
            linked_doses = plan_to_doses.get(plan.sop_instance_uid, [])
            preferred_dose = None
            if linked_doses:
                plan_doses = [dose for dose in linked_doses if dose.dose_summation_type.upper() == "PLAN"]
                preferred_dose = plan_doses[0] if plan_doses else linked_doses[0]
            struct_uids = split_refs(plan.referenced_rtstruct_uid)
            linked_struct = structs.get(struct_uids[0]) if len(struct_uids) == 1 else None
            warnings: list[str] = []
            if len(linked_doses) > 1:
                warnings.append("multiple_rtdose")
            if len(struct_uids) > 1:
                warnings.append("multiple_rtstruct")
            frame_uid = plan.frame_of_reference_uid or (linked_struct.frame_of_reference_uid if linked_struct else "") or (preferred_dose.frame_of_reference_uid if preferred_dose else "")
            course_type = detect_course_type(plan, warnings)
            courses.append(
                RtCourse(
                    patient_id=patient_id,
                    rt_course_id=f"{patient_id}_course_{index:02d}",
                    plan=plan,
                    dose=preferred_dose,
                    struct=linked_struct,
                    course_date=plan.rtplan_date or plan.study_date,
                    frame_of_reference_uid=frame_uid,
                    course_type=course_type,
                    warning=seq_text(dict.fromkeys(warnings)),
                )
            )

        linked_plan_uids = {plan.sop_instance_uid for plan in patient_plans}
        standalone_doses = [dose for dose in patient_doses if not (set(split_refs(dose.referenced_rtplan_uid)) & linked_plan_uids)]
        for dose in standalone_doses:
            index = len([course for course in courses if course.patient_id == patient_id]) + 1
            courses.append(
                RtCourse(
                    patient_id=patient_id,
                    rt_course_id=f"{patient_id}_course_{index:02d}",
                    plan=None,
                    dose=dose,
                    struct=None,
                    course_date=dose.study_date,
                    frame_of_reference_uid=dose.frame_of_reference_uid,
                    course_type="dose_only",
                    warning="dose_without_plan",
                )
            )

        patient_course_counts[patient_id] = len([course for course in courses if course.patient_id == patient_id])
        if patient_course_counts[patient_id] > 1:
            dated_courses = sorted(
                [course for course in courses if course.patient_id == patient_id],
                key=lambda course: parse_date(course.course_date) or datetime.max,
            )
            if len(dated_courses) >= 2:
                delta_days = None
                first_date = parse_date(dated_courses[0].course_date)
                last_date = parse_date(dated_courses[-1].course_date)
                if first_date and last_date:
                    delta_days = (last_date - first_date).days
                if delta_days is not None and delta_days > 120:
                    for course in dated_courses[1:]:
                        course.warning = seq_text(dict.fromkeys(split_refs(course.warning) + ["possible_reirradiation"]))
                        if course.course_type == "initial_or_unspecified":
                            course.course_type = "possible_reirradiation"

    return courses, patient_course_counts


def range_from_values(values: list[float]) -> tuple[float, float] | None:
    """Return min/max range when values exist."""
    return (min(values), max(values)) if values else None


def spacing_tuple(text: str) -> tuple[float, float] | None:
    """Parse PixelSpacing text."""
    values = [safe_float(item) for item in text.split("|") if item]
    if len(values) >= 2 and values[0] is not None and values[1] is not None:
        return values[0], values[1]
    return None


def ct_bbox(info: CtSeriesInfo) -> dict[str, Any]:
    """Approximate CT physical bounding box from image positions and spacing."""
    if not info.positions:
        return {"z_range": None, "bbox": None, "coverage_mm": None}
    spacing = spacing_tuple(most_common(info.pixel_spacings))
    rows = most_common(info.rows)
    cols = most_common(info.columns)
    xs = [pos[0] for pos in info.positions]
    ys = [pos[1] for pos in info.positions]
    zs = [pos[2] for pos in info.positions]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    if spacing and rows and cols:
        x_max += cols * spacing[1]
        y_max += rows * spacing[0]
    z_min = min(zs)
    z_max = max(zs)
    coverage_mm = abs(z_max - z_min)
    return {
        "z_range": (z_min, z_max),
        "bbox": ((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        "coverage_mm": coverage_mm,
    }


def dose_bbox(dose: DoseInfo | None) -> dict[str, Any]:
    """Approximate RTDOSE physical bounding box."""
    if dose is None or dose.image_position is None or dose.pixel_spacing is None:
        return {"z_range": None, "bbox": None, "center": None}
    x0, y0, z0 = dose.image_position
    row_spacing, col_spacing = dose.pixel_spacing
    x1 = x0 + (dose.columns or 0) * col_spacing
    y1 = y0 + (dose.rows or 0) * row_spacing
    z_values = [z0 + offset for offset in dose.z_offsets] if dose.z_offsets else [z0]
    z_range = range_from_values(z_values)
    if z_range is None:
        return {"z_range": None, "bbox": None, "center": None}
    bbox = ((min(x0, x1), max(x0, x1)), (min(y0, y1), max(y0, y1)), z_range)
    center = (
        (bbox[0][0] + bbox[0][1]) / 2.0,
        (bbox[1][0] + bbox[1][1]) / 2.0,
        (bbox[2][0] + bbox[2][1]) / 2.0,
    )
    return {"z_range": z_range, "bbox": bbox, "center": center}


def overlap_fraction(ct_range: tuple[float, float] | None, dose_range_value: tuple[float, float] | None) -> float:
    """Compute 1D overlap fraction over the dose range."""
    if ct_range is None or dose_range_value is None:
        return 0.0
    start = max(min(ct_range), min(dose_range_value))
    end = min(max(ct_range), max(dose_range_value))
    if end <= start:
        return 0.0
    dose_span = abs(dose_range_value[1] - dose_range_value[0])
    if dose_span == 0:
        return 1.0 if start == end else 0.0
    return max(0.0, min(1.0, (end - start) / dose_span))


def point_inside_bbox(point: tuple[float, float, float] | None, bbox_value: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None) -> bool:
    """Test whether a point lies inside an axis-aligned bounding box."""
    if point is None or bbox_value is None:
        return False
    return (
        bbox_value[0][0] <= point[0] <= bbox_value[0][1]
        and bbox_value[1][0] <= point[1] <= bbox_value[1][1]
        and bbox_value[2][0] <= point[2] <= bbox_value[2][1]
    )


def path_keyword_score(flags: dict[str, bool]) -> int:
    """Score positive path keywords."""
    score = 0
    if flags["TRA"] and flags["LUNG"] and flags["CT"]:
        score += 8
    for keyword in ("THORAX", "PLAN", "PL", "BPL", "MAMMA"):
        if flags[keyword]:
            score += 2
    if flags["BOOST"] or flags["BST"]:
        score += 1
    return score


def kernel_soft_tissue_score(kernel: str) -> int:
    """Score kernel compatibility for soft tissue planning CT."""
    kernel_upper = safe_str(kernel).upper()
    if not kernel_upper:
        return 0
    if any(token in kernel_upper for token in ("B70", "B80", "LUNG", "SHARP")):
        return -4
    if any(token in kernel_upper for token in ("B20", "B30", "STANDARD", "SOFT", "BODY", "MED")):
        return 6
    return 1


def is_negative_candidate(info: CtSeriesInfo) -> list[str]:
    """Detect negative series evidence."""
    texts = [info.relative_path, info.series_description, info.protocol_name, info.study_description]
    return negative_match(texts)


def build_inventory_rows(ct_series: dict[str, CtSeriesInfo], courses: list[RtCourse]) -> list[dict[str, Any]]:
    """Build folder-pattern inventory rows."""
    rows: list[dict[str, Any]] = []
    courses_by_patient: dict[str, list[RtCourse]] = defaultdict(list)
    for course in courses:
        courses_by_patient[course.patient_id].append(course)

    for info in sorted(ct_series.values(), key=lambda item: (item.patient_id, item.exam_date, item.relative_path)):
        patient_courses = courses_by_patient.get(info.patient_id, [None])
        if not patient_courses:
            patient_courses = [None]
        for course in patient_courses:
            dose_meta = dose_bbox(course.dose) if course else {"z_range": None, "bbox": None, "center": None}
            ct_meta = ct_bbox(info)
            flags = keyword_flags([info.exam_folder_name, info.relative_path, info.series_description, info.protocol_name, info.study_description])
            struct_ref_match = False
            frame_match = False
            slice_equal = False
            slice_diff = None
            if course and course.struct:
                struct_ref_match = info.series_instance_uid in split_refs(course.struct.referenced_ct_series_uid)
            if course and course.frame_of_reference_uid:
                frame_match = info.frame_of_reference_uid == course.frame_of_reference_uid
            if course and course.dose and course.dose.number_of_frames is not None:
                slice_diff = abs(info.number_of_ct_slices - course.dose.number_of_frames)
                slice_equal = info.number_of_ct_slices == course.dose.number_of_frames
            overlap = overlap_fraction(ct_meta["z_range"], dose_meta["z_range"])
            rows.append(
                {
                    "patient_id": info.patient_id,
                    "rt_course_id": course.rt_course_id if course else "",
                    "patient_folder_name": info.patient_id,
                    "examination_date_folder_name": info.exam_folder_name,
                    "date_folder_contains_rt": flags["RT"],
                    "parsed_examination_date": info.exam_date,
                    "study_date": info.study_date,
                    "ct_series_relative_path": info.relative_path,
                    "path_depth": info.path_depth,
                    "ct_series_folder_name": info.series_folder_name,
                    "keyword_rt": flags["RT"],
                    "keyword_tra": flags["TRA"],
                    "keyword_lung": flags["LUNG"],
                    "keyword_ct": flags["CT"],
                    "keyword_thorax": flags["THORAX"],
                    "keyword_plan": flags["PLAN"],
                    "keyword_pl": flags["PL"],
                    "keyword_bpl": flags["BPL"],
                    "keyword_mamma": flags["MAMMA"],
                    "keyword_boost": flags["BOOST"],
                    "keyword_bst": flags["BST"],
                    "ct_modality": "CT",
                    "series_description": info.series_description,
                    "protocol_name": info.protocol_name,
                    "study_description": info.study_description,
                    "convolution_kernel": most_common(info.convolution_kernels),
                    "slice_thickness_mm": most_common(info.slice_thicknesses),
                    "spacing_between_slices_mm": most_common(info.spacing_between_slices),
                    "pixel_spacing": most_common(info.pixel_spacings),
                    "number_of_ct_slices": info.number_of_ct_slices,
                    "z_axis_physical_coverage_mm": ct_meta["coverage_mm"],
                    "frame_of_reference_uid": info.frame_of_reference_uid,
                    "study_instance_uid": info.study_instance_uid,
                    "series_instance_uid": info.series_instance_uid,
                    "referenced_by_rtstruct": struct_ref_match,
                    "shares_rtdose_frame_of_reference_uid": frame_match,
                    "ct_slice_count_equals_rtdose_frames": slice_equal,
                    "ct_rtdose_slice_count_difference": slice_diff,
                    "ct_rtdose_z_range_overlap": overlap > 0,
                    "rtdose_overlap_fraction": overlap,
                    "dose_center_inside_ct": point_inside_bbox(dose_meta["center"], ct_meta["bbox"]),
                }
            )
    return rows


def rank_candidates(ct_series: dict[str, CtSeriesInfo], courses: list[RtCourse], patient_course_counts: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank planning CT candidates per RT course."""
    courses_by_patient: dict[str, list[RtCourse]] = defaultdict(list)
    series_by_patient: dict[str, list[CtSeriesInfo]] = defaultdict(list)
    for course in courses:
        courses_by_patient[course.patient_id].append(course)
    for info in ct_series.values():
        series_by_patient[info.patient_id].append(info)

    ranking_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    multi_course_rows: list[dict[str, Any]] = []

    for patient_id, patient_courses in sorted(courses_by_patient.items()):
        if patient_course_counts.get(patient_id, 0) > 1:
            for course in patient_courses:
                multi_course_rows.append(
                    {
                        "patient_id": patient_id,
                        "rt_course_id": course.rt_course_id,
                        "course_date": course.course_date,
                        "course_type": course.course_type,
                        "plan_uid": course.plan.sop_instance_uid if course.plan else "",
                        "dose_uid": course.dose.sop_instance_uid if course.dose else "",
                        "struct_uid": course.struct.sop_instance_uid if course.struct else "",
                        "frame_of_reference_uid": course.frame_of_reference_uid,
                        "warning": course.warning,
                    }
                )

        sorted_courses = sorted(patient_courses, key=lambda course: parse_date(course.course_date) or datetime.max)
        earliest_course_id = sorted_courses[0].rt_course_id if sorted_courses else ""

        for course in sorted_courses:
            candidates: list[dict[str, Any]] = []
            dose_meta = dose_bbox(course.dose)
            for info in sorted(series_by_patient.get(patient_id, []), key=lambda item: (item.exam_date, item.relative_path)):
                flags = keyword_flags([info.exam_folder_name, info.relative_path, info.series_description, info.protocol_name, info.study_description])
                negatives = is_negative_candidate(info)
                ct_meta = ct_bbox(info)
                overlap = overlap_fraction(ct_meta["z_range"], dose_meta["z_range"])
                dose_center_inside = point_inside_bbox(dose_meta["center"], ct_meta["bbox"])
                struct_ref_match = bool(course.struct and info.series_instance_uid in split_refs(course.struct.referenced_ct_series_uid))
                frame_match = bool(course.frame_of_reference_uid and info.frame_of_reference_uid == course.frame_of_reference_uid)
                slice_diff = abs(info.number_of_ct_slices - (course.dose.number_of_frames or info.number_of_ct_slices)) if course.dose and course.dose.number_of_frames is not None else None
                slice_equal = slice_diff == 0 if slice_diff is not None else False
                earliest_rt_bonus = 2 if course.rt_course_id == earliest_course_id and flags["RT"] and info.exam_date else 0
                coverage_mm = ct_meta["coverage_mm"] or 0.0

                score = 0
                reasons: list[str] = []
                warnings: list[str] = []

                if struct_ref_match:
                    score += 120
                    reasons.append("rtstruct_reference_match")
                if frame_match:
                    score += 90
                    reasons.append("frame_of_reference_match")
                elif course.frame_of_reference_uid:
                    score -= 120
                    warnings.append("frame_of_reference_mismatch")

                if course.dose and course.plan and course.plan.sop_instance_uid in split_refs(course.dose.referenced_rtplan_uid):
                    score += 30
                    reasons.append("dose_plan_link")
                if course.plan and course.struct and course.struct.sop_instance_uid in split_refs(course.plan.referenced_rtstruct_uid):
                    score += 30
                    reasons.append("plan_struct_link")

                if overlap > 0:
                    score += int(round(overlap * 60))
                    reasons.append("physical_overlap")
                elif course.dose:
                    score -= 150
                    warnings.append("no_physical_overlap")

                if dose_center_inside:
                    score += 25
                    reasons.append("dose_center_inside_ct")

                if flags["RT"]:
                    score += 10
                    reasons.append("rt_folder_keyword")
                keyword_score = path_keyword_score(flags)
                score += keyword_score
                if keyword_score:
                    reasons.append("thoracic_path_keywords")

                slice_thickness = safe_float(most_common(info.slice_thicknesses))
                if slice_thickness is not None and 2.0 <= slice_thickness <= 5.0:
                    score += 8
                    reasons.append("slice_thickness_2_to_5mm")

                score += kernel_soft_tissue_score(most_common(info.convolution_kernels))
                if kernel_soft_tissue_score(most_common(info.convolution_kernels)) > 0:
                    reasons.append("soft_tissue_kernel")
                if coverage_mm >= 200:
                    score += 8
                    reasons.append("thoracic_coverage")
                elif coverage_mm and coverage_mm < 120:
                    score -= 15
                    warnings.append("incomplete_thoracic_coverage")

                if slice_equal:
                    score += 6
                    reasons.append("slice_count_equal_to_dose_frames")
                elif slice_diff is not None and slice_diff <= 3:
                    score += 3
                    reasons.append("slice_count_close_to_dose_frames")

                score += earliest_rt_bonus
                if earliest_rt_bonus:
                    reasons.append("earliest_rt_folder_low_weight")

                if negatives:
                    score -= 60
                    warnings.append(seq_text(negatives))

                candidate = {
                    "patient_id": patient_id,
                    "rt_course_id": course.rt_course_id,
                    "candidate_ct_series_uid": info.series_instance_uid,
                    "candidate_ct_relative_path": info.relative_path,
                    "study_date": info.study_date,
                    "series_description": info.series_description,
                    "protocol_name": info.protocol_name,
                    "convolution_kernel": most_common(info.convolution_kernels),
                    "slice_thickness_mm": slice_thickness,
                    "number_of_ct_slices": info.number_of_ct_slices,
                    "rtdose_number_of_frames": course.dose.number_of_frames if course.dose else None,
                    "slice_count_equal": slice_equal,
                    "slice_count_difference": slice_diff,
                    "rt_folder_keyword_score": 10 if flags["RT"] else 0,
                    "path_keyword_score": keyword_score,
                    "rtstruct_reference_match": struct_ref_match,
                    "frame_of_reference_match": frame_match,
                    "physical_overlap_fraction": overlap,
                    "dose_center_inside_ct": dose_center_inside,
                    "candidate_score": score,
                    "selection_reason": seq_text(dict.fromkeys(reasons)),
                    "warning": seq_text(dict.fromkeys(warnings + split_refs(course.warning))),
                    "exam_date": info.exam_date,
                    "date_folder_contains_rt": flags["RT"],
                    "series_name_contains_plan_keywords": any(flags[key] for key in ("THORAX", "PLAN", "PL", "BPL")),
                    "path_contains_tra_lung_ct": flags["TRA"] and flags["LUNG"] and flags["CT"],
                    "frame_of_reference_uid": info.frame_of_reference_uid,
                }
                candidates.append(candidate)

            candidates.sort(key=lambda row: (-row["candidate_score"], not row["rtstruct_reference_match"], not row["frame_of_reference_match"], row["slice_count_difference"] if row["slice_count_difference"] is not None else 999999, row["candidate_ct_relative_path"]))
            for rank, candidate in enumerate(candidates, start=1):
                candidate["candidate_rank"] = rank
                candidate["selection_status"] = "candidate"

            if candidates:
                top = candidates[0]
                second_score = candidates[1]["candidate_score"] if len(candidates) > 1 else None
                manual_review = False
                if "frame_of_reference_mismatch" in top["warning"] or "no_physical_overlap" in top["warning"]:
                    manual_review = True
                if second_score is not None and abs(top["candidate_score"] - second_score) <= 10:
                    manual_review = True
                if not top["rtstruct_reference_match"] and not top["frame_of_reference_match"] and top["physical_overlap_fraction"] == 0:
                    manual_review = True

                top["selection_status"] = "manual_review" if manual_review else "selected"
                selected_rows.append(top.copy())
                if manual_review:
                    manual_rows.append(top.copy())
                if len(candidates) > 1 and top["rtstruct_reference_match"] and not candidates[1]["rtstruct_reference_match"]:
                    top["selection_reason"] = seq_text(split_refs(top["selection_reason"]) + ["unique_rtstruct_reference"])

            ranking_rows.extend(candidates)

    return ranking_rows, selected_rows, multi_course_rows, manual_rows


def summary_metrics(selected_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]], multi_course_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build cohort-level summary metrics."""
    if not selected_rows:
        return []
    total = len(selected_rows)
    thickness_counter = Counter(str(row["slice_thickness_mm"]) for row in selected_rows if row["slice_thickness_mm"] is not None)
    kernel_counter = Counter(row["convolution_kernel"] for row in selected_rows if row["convolution_kernel"])
    metrics = [
        ("selected_cases", total),
        ("pct_in_rt_folder", round(100.0 * sum(1 for row in selected_rows if row["date_folder_contains_rt"]) / total, 2)),
        ("pct_in_earliest_rt_related_date_folder", round(100.0 * sum(1 for row in selected_rows if "earliest_rt_folder_low_weight" in row["selection_reason"]) / total, 2)),
        ("pct_with_TRA_LUNG_CT_path", round(100.0 * sum(1 for row in selected_rows if row["path_contains_tra_lung_ct"]) / total, 2)),
        ("pct_with_THORAX_PLAN_PL_BPL_name", round(100.0 * sum(1 for row in selected_rows if row["series_name_contains_plan_keywords"]) / total, 2)),
        ("slice_thickness_distribution", seq_text(f"{key}:{value}" for key, value in thickness_counter.most_common())),
        ("convolution_kernel_distribution", seq_text(f"{key}:{value}" for key, value in kernel_counter.most_common())),
        ("pct_slice_count_equal_to_dose_frames", round(100.0 * sum(1 for row in selected_rows if row["slice_count_equal"]) / total, 2)),
        ("pct_uniquely_identified_by_rtstruct_reference", round(100.0 * sum(1 for row in selected_rows if "unique_rtstruct_reference" in row["selection_reason"]) / total, 2)),
        ("pct_identified_by_frame_of_reference_and_geometry", round(100.0 * sum(1 for row in selected_rows if row["frame_of_reference_match"] and row["physical_overlap_fraction"] > 0 and not row["rtstruct_reference_match"]) / total, 2)),
        ("n_cases_requiring_manual_review", len(manual_rows)),
        ("n_patients_with_multiple_rt_courses", len({row["patient_id"] for row in multi_course_rows})),
        ("n_possible_boost_or_reirradiation_cases", sum(1 for row in multi_course_rows if "boost" in row["course_type"] or "reirradiation" in row["course_type"])),
    ]
    return [{"metric": key, "value": value} for key, value in metrics]


def write_report(path: Path, summary_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> None:
    """Write human-readable markdown report."""
    summary_map = {row["metric"]: row["value"] for row in summary_rows}
    lines = [
        "# MINT Folder Pattern Report",
        "",
        "## Cohort Summary",
        f"- Selected planning CT cases: {summary_map.get('selected_cases', 0)}",
        f"- In folders containing `RT`: {summary_map.get('pct_in_rt_folder', 0)}%",
        f"- In earliest RT-related date folder: {summary_map.get('pct_in_earliest_rt_related_date_folder', 0)}%",
        f"- Paths containing `TRA/LUNG/CT`: {summary_map.get('pct_with_TRA_LUNG_CT_path', 0)}%",
        f"- Names containing `THORAX`/`PLAN`/`PL`/`BPL`: {summary_map.get('pct_with_THORAX_PLAN_PL_BPL_name', 0)}%",
        f"- Slice-count equals RTDOSE frames: {summary_map.get('pct_slice_count_equal_to_dose_frames', 0)}%",
        f"- Uniquely identified through RTSTRUCT reference: {summary_map.get('pct_uniquely_identified_by_rtstruct_reference', 0)}%",
        f"- Identified by FrameOfReferenceUID plus geometry: {summary_map.get('pct_identified_by_frame_of_reference_and_geometry', 0)}%",
        f"- Manual review cases: {summary_map.get('n_cases_requiring_manual_review', 0)}",
        f"- Patients with multiple RT courses: {summary_map.get('n_patients_with_multiple_rt_courses', 0)}",
        f"- Possible boost/reirradiation cases: {summary_map.get('n_possible_boost_or_reirradiation_cases', 0)}",
        "",
        "## Reliable Patterns",
        "- RTSTRUCT-referenced CT series is the strongest cohort-wide planning-CT indicator when present.",
        "- Matching FrameOfReferenceUID plus non-zero RTDOSE overlap is a strong fallback when RTSTRUCT linkage is absent.",
        "- Folder names containing `RT` and thoracic path tokens are helpful but not sufficient on their own.",
        "- CT slice-count matching RTDOSE frames is supportive only and should not be treated as definitive linkage.",
        "",
        "## Manual Review Focus",
        "- Review cases with FrameOfReference mismatch, zero physical overlap, or near-tied candidate scores.",
        "- Multiple RT courses, boost plans, and widely separated plan dates remain high-risk for incorrect automatic selection.",
        "",
        "## Selected Examples",
    ]
    for row in selected_rows[:10]:
        lines.append(
            f"- {row['patient_id']} {row['rt_course_id']}: rank {row['candidate_rank']} score {row['candidate_score']} path `{row['candidate_ct_relative_path']}` reason `{row['selection_reason']}`"
        )
    if manual_rows:
        lines.extend(["", "## Manual Review Cases"])
        for row in manual_rows[:20]:
            lines.append(
                f"- {row['patient_id']} {row['rt_course_id']}: `{row['candidate_ct_relative_path']}` warning `{row['warning']}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV output."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Standalone ranking workflow."""
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_source(source_dir)
    ct_series = build_ct_series(records)
    courses, patient_course_counts = group_rt_courses(records)
    inventory_rows = build_inventory_rows(ct_series, courses)
    ranking_rows, selected_rows, multi_course_rows, manual_rows = rank_candidates(ct_series, courses, patient_course_counts)
    summary_rows = summary_metrics(selected_rows, manual_rows, multi_course_rows)

    if args.dry_run:
        print(f"ct_series: {len(ct_series)}")
        print(f"rt_courses: {len(courses)}")
        print(f"candidate_rows: {len(ranking_rows)}")
        print(f"selected_rows: {len(selected_rows)}")
        print(f"manual_review_rows: {len(manual_rows)}")
        return 0

    write_csv(output_dir / "mint_folder_pattern_inventory.csv", inventory_rows)
    write_csv(output_dir / "mint_ct_candidate_ranking.csv", ranking_rows)
    write_csv(output_dir / "mint_folder_pattern_summary.csv", summary_rows)
    write_csv(output_dir / "mint_planning_ct_selection_qc.csv", selected_rows)
    write_csv(output_dir / "mint_multiple_rt_course_cases.csv", multi_course_rows)
    write_csv(output_dir / "mint_manual_review_cases.csv", manual_rows)
    write_report(output_dir / "mint_folder_pattern_report.md", summary_rows, selected_rows, manual_rows)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
