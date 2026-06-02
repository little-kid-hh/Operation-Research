param(
    [ValidateSet("label_2ori", "label_6ori")]
    [string]$Target = "label_6ori",

    [int]$Splits = 5,

    [ValidateSet("order", "instance")]
    [string]$GroupLevel = "instance",

    [switch]$Quick,

    [string[]]$Models = @(),

    [string]$Python = "D:\python_anaconda\python.exe",

    [string]$TempDir = ".codex_training_tmp",

    [string]$DataPath = "permin_dataset_processing\processed_features\or2023_bpp_labeled_base40_package.csv"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:JOBLIB_TEMP_FOLDER = $TempDir

if (Test-Path ".codex_xgb_only") {
    $env:PYTHONPATH = ".codex_xgb_only"
}

$argsList = @(
    "permin_dataset_processing\cross_validate_permin_models.py",
    "--data-path", $DataPath,
    "--target", $Target,
    "--group-level", $GroupLevel,
    "--n-splits", "$Splits"
)

if ($Quick) {
    $argsList += "--quick"
}

if ($Models.Count -gt 0) {
    $argsList += "--models"
    $argsList += $Models
}

& $Python @argsList
