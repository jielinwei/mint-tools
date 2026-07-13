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

- DICOM conversion
- Segmentation
- Dose resampling
- Radiomics extraction
- Any modification of raw source data

## Repository layout

- `scripts/audit_mint_rt_dicom.py`: base RT DICOM audit
- `scripts/analyze_mint_folder_patterns.py`: planning-CT pattern analysis
- `scripts/rank_planning_ct_candidates.py`: candidate ranking logic
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
- `logs/audit_mint_rt_dicom.log`
