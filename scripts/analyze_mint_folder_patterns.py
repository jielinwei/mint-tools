#!/usr/bin/env python3
"""Analyze MINT folder patterns and planning CT candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rank_planning_ct_candidates import (
    build_ct_series,
    build_inventory_rows,
    group_rt_courses,
    rank_candidates,
    scan_source,
    summary_metrics,
    write_csv,
    write_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run folder pattern analysis."""
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_source(source_dir)
    ct_series = build_ct_series(records)
    courses, patient_course_counts = group_rt_courses(records)
    inventory_rows = build_inventory_rows(ct_series, courses)
    ranking_rows, selected_rows, multi_course_rows, manual_rows = rank_candidates(
        ct_series, courses, patient_course_counts
    )
    summary_rows = summary_metrics(selected_rows, manual_rows, multi_course_rows)

    if args.dry_run:
        print("Dry-run output targets:")
        for name in (
            "mint_folder_pattern_inventory.csv",
            "mint_ct_candidate_ranking.csv",
            "mint_folder_pattern_summary.csv",
            "mint_planning_ct_selection_qc.csv",
            "mint_multiple_rt_course_cases.csv",
            "mint_manual_review_cases.csv",
            "mint_folder_pattern_report.md",
        ):
            print(f"- {output_dir / name}")
        print(f"dicom_records: {len(records)}")
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
    raise SystemExit(main())
