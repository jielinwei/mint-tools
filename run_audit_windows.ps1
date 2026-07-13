param(
    [string]$SourceDir,
    [string]$OutputDir = ".",
    [switch]$DryRun,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

if (-not $SourceDir) {
    Write-Error "SourceDir is required."
}

$auditArgs = @(
    "scripts/audit_mint_rt_dicom.py",
    "--source-dir", $SourceDir,
    "--output-dir", $OutputDir
)

$patternArgs = @(
    "scripts/analyze_mint_folder_patterns.py",
    "--source-dir", $SourceDir,
    "--output-dir", (Join-Path $OutputDir "outputs")
)

if ($DryRun) {
    $auditArgs += "--dry-run"
    $patternArgs += "--dry-run"
}

if ($Overwrite) {
    $auditArgs += "--overwrite"
}

python @auditArgs
python @patternArgs
