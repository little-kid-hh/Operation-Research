param(
    [int]$BatchSize = 10000,
    [string]$OutputPath = "permin_dataset_processing\milp_labels\or2023_bpp_package_labels.csv",
    [Parameter(Mandatory = $true)]
    [string]$XmlDir,
    [Parameter(Mandatory = $true)]
    [string]$PackagesPath,
    [int]$TotalTasks,
    [double]$TimeLimit2Ori = 300.0,
    [double]$TimeLimit6Ori = 300.0,
    [switch]$AllowBspDerivedData
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path "."
$classPath = "MILP_3DBPP\target\classes;C:\gurobi1300\win64\lib\gurobi.jar"
$tmpDir = "permin_dataset_processing\milp_labels\batches"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$xmlNameRegex = ".*\.xml$"
$allowBspArg = if ($AllowBspDerivedData) { "true" } else { "false" }

if (!$AllowBspDerivedData -and (($XmlDir -replace "\\", "/") -like "*S3DBSP-main*" -or ($PackagesPath -replace "\\", "/") -like "*S3DBSP-main*")) {
    throw "Refusing S3DBSP/stochastic-BSP data path. Use the Fontaine & Minner OR 2023 3D-BPP e-companion data path, or pass -AllowBspDerivedData only for legacy audits."
}

if ($TotalTasks -le 0) {
    throw "TotalTasks is required for OR 2023 BPP batch labeling. Compute orders x candidate packages for the selected data first."
}

function Get-CompletedTasks {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return 0
    }
    $lineCount = (Get-Content $Path | Measure-Object -Line).Lines
    return [Math]::Max(0, $lineCount - 1)
}

$completed = Get-CompletedTasks -Path $OutputPath
Write-Host "Existing completed tasks: $completed / $TotalTasks"

while ($completed -lt $TotalTasks) {
    $remaining = $TotalTasks - $completed
    $currentBatchSize = [Math]::Min($BatchSize, $remaining)
    $batchOutput = Join-Path $tmpDir ("batch_{0:D7}_{1:D7}.csv" -f $completed, ($completed + $currentBatchSize - 1))
    $stdoutPath = "$batchOutput.out.log"
    $stderrPath = "$batchOutput.err.log"

    Write-Host "Running batch start=$completed size=$currentBatchSize"
    java `
        "-Dor2023.bpp.timeLimit2ori=$TimeLimit2Ori" `
        "-Dor2023.bpp.timeLimit6ori=$TimeLimit6Ori" `
        -cp $classPath org.example.GeneratePerminPackageLabels `
        $XmlDir `
        $PackagesPath `
        $batchOutput `
        -1 `
        -1 `
        -1 `
        $xmlNameRegex `
        $completed `
        $currentBatchSize `
        $allowBspArg `
        > $stdoutPath 2> $stderrPath

    if ($LASTEXITCODE -ne 0) {
        throw "Java label batch failed with exit code $LASTEXITCODE. See $stderrPath"
    }

    $batchLines = (Get-Content $batchOutput | Measure-Object -Line).Lines
    $batchTasks = [Math]::Max(0, $batchLines - 1)
    if ($batchTasks -le 0) {
        throw "Batch produced no tasks: $batchOutput"
    }

    if (!(Test-Path $OutputPath)) {
        Copy-Item $batchOutput $OutputPath
    } else {
        Get-Content $batchOutput | Select-Object -Skip 1 | Add-Content $OutputPath
    }

    $completed = Get-CompletedTasks -Path $OutputPath
    Write-Host "Completed tasks: $completed / $TotalTasks"
}

Write-Host "All tasks completed: $OutputPath"
