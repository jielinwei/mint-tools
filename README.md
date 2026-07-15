# MINT Tools

Python tools for audit-only inspection of radiotherapy DICOM datasets, with emphasis on planning CT selection and RT object linkage.

## Scope

- Recursively scan a source directory
- Detect DICOM with `pydicom`
- Classify CT / RTDOSE / RTPLAN / RTSTRUCT / MR / other
- Build patient-study-series-frame linkage tables
- Identify candidate linked radiotherapy cases
- Analyze folder naming patterns for planning CT selection
- Export CSV summaries for manual review

Out of scope:

- Segmentation
- Dose resampling
- Radiomics extraction
- Any modification of raw source data

DICOM conversion is intentionally excluded from the audit stage. After manual planning CT resolution, `scripts/export_planning_ct_nifti.py` can export the confirmed CT series to NIfTI without modifying raw DICOM files.

## Repository layout

- `scripts/audit_mint_rt_dicom.py`: base RT DICOM audit
- `scripts/analyze_mint_folder_patterns.py`: planning-CT pattern analysis
- `scripts/rank_planning_ct_candidates.py`: candidate ranking logic
- `scripts/export_planning_ct_nifti.py`: export resolved planning CT DICOM series to NIfTI
- `config.example.yaml`: example configuration
- `requirements.txt`: Python dependencies
- `run_audit_windows.ps1`: Windows PowerShell entrypoint
- `run_audit_windows.bat`: Windows batch entrypoint

Generated directories such as `outputs/`, `logs/`, and `__pycache__/` are intentionally excluded from git.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit the paths for your environment.

Raw data rule:

- treat raw data as read-only
- do not modify, move, delete, or copy raw data

## Usage

Base audit dry-run:

```bash
python scripts/audit_mint_rt_dicom.py --source-dir "/path/to/raw_data" --output-dir . --dry-run
```

Base audit full run:

```bash
python scripts/audit_mint_rt_dicom.py --source-dir "/path/to/raw_data" --output-dir . --overwrite
```

Folder-pattern dry-run:

```bash
python scripts/analyze_mint_folder_patterns.py --source-dir "/path/to/raw_data" --output-dir outputs --dry-run
```

Folder-pattern full run:

```bash
python scripts/analyze_mint_folder_patterns.py --source-dir "/path/to/raw_data" --output-dir outputs
```

Export resolved planning CT to NIfTI:

```bash
python scripts/export_planning_ct_nifti.py \
  --source-dir "/path/to/raw_data" \
  --resolved-csv "outputs/mint_manual_review_final_planning_ct_resolved.csv" \
  --output-dir "outputs/planningCT" \
  --dry-run

python scripts/export_planning_ct_nifti.py \
  --source-dir "/path/to/raw_data" \
  --resolved-csv "outputs/mint_manual_review_final_planning_ct_resolved.csv" \
  --output-dir "outputs/planningCT"
```

The exporter writes one file per unique planning CT using the name pattern `patientID_YYYYMMDD_planningCT.nii.gz`. When `final_ct_series_uid` is available, it aggregates all CT DICOM files with the same SeriesInstanceUID across the full patient folder, because some MINT CT series are split across multiple source folders.

Knowledge-table filter:

```bash
python scripts/audit_mint_rt_dicom.py --source-dir "/path/to/raw_data" --output-dir . --dry-run --knowledge-xlsx "/path/to/folder_structure3_categeried2026_RT_RT232_matched2.xlsx" --knowledge-sheet "RT243-3012-final"
python scripts/analyze_mint_folder_patterns.py --source-dir "/path/to/raw_data" --output-dir outputs --dry-run --knowledge-xlsx "/path/to/folder_structure3_categeried2026_RT_RT232_matched2.xlsx" --knowledge-sheet "RT243-3012-final"
```

The filter keeps only files under source folders matching an `ID` and `Date` pair in the selected Excel sheet. The expected source-folder pattern is `<ID>/<Date>/...`, where `Date` is the second-level examination folder, for example `20130710-RTLUNG_MAMMA-CUR-BST`.

Optional controls:

- `--max-patients N`
- `--overwrite`

## Main outputs

- `outputs/mint_dicom_inventory.csv`
- `outputs/mint_patient_summary.csv`
- `outputs/mint_rt_case_summary.csv`
- `outputs/mint_rt_linkage_qc.csv`
- `outputs/mint_rt_exclusion_candidates.csv`
- `outputs/mint_folder_pattern_inventory.csv`
- `outputs/mint_ct_candidate_ranking.csv`
- `outputs/mint_folder_pattern_summary.csv`
- `outputs/mint_planning_ct_selection_qc.csv`
- `outputs/mint_multiple_rt_course_cases.csv`
- `outputs/mint_manual_review_cases.csv`
- `outputs/planningCT/*_planningCT.nii.gz`
- `outputs/planningCT/planningCT_export_qc.csv`
- `logs/audit_mint_rt_dicom.log`
