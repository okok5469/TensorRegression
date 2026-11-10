param(
    [ValidateSet('prepare', 'run')][string]$Action = 'prepare',
    [ValidateSet('lemon', 'hcp_ya', 'nki_rs')][string]$Dataset = 'lemon',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $repo
if (-not $Python) { $Python = Join-Path $repo '.venv/Scripts/python.exe' }
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $Python -B -m research.taskcard12.empirical $Action --config "research/taskcard12/empirical_configs/$Dataset.json"
exit $LASTEXITCODE
