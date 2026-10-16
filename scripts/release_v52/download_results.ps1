param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [string]$RemoteRoot = '/public/home/xjh/xmx/tensor-sparse-regression-main',
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
if ($SshHost -notmatch '^[A-Za-z0-9_.@:-]+$' -or $SshHost.StartsWith('-')) { throw 'Invalid SSH host' }
if (-not $RemoteRoot.StartsWith('/') -or $RemoteRoot -match '[\r\n\x00]') { throw 'Invalid remote root' }
$taskFiles = @('raw_results.csv','summary_results.csv','integrity_report.json','plan.json')
$taskRemote = $RemoteRoot + '/numerical_simulation/outputs_v52_search_final_full/'
'DRY_RUN: merged summary files only; no NPZ download'
$taskFiles | ForEach-Object { $taskRemote + $_ }
if (-not $Apply) { return }
$taskDestination = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $taskDestination) { throw 'Use a new destination; existing results are never overwritten' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
foreach ($taskFile in $taskFiles) {
    $taskPath = $taskRemote + $taskFile
    & scp ($SshHost + ':' + $taskPath) (Join-Path $taskDestination $taskFile)
    if ($LASTEXITCODE -ne 0) { throw 'Download failed; retain partial directory for inspection' }
}
$taskReport = Get-Content -LiteralPath (Join-Path $taskDestination 'integrity_report.json') -Raw | ConvertFrom-Json
$taskPlan = Get-Content -LiteralPath (Join-Path $taskDestination 'plan.json') -Raw | ConvertFrom-Json
if ($taskReport.integrity_checks_passed -ne $true -or $taskReport.task_count -ne 700 -or
    $taskReport.method_result_count -ne 2800 -or $taskReport.plan_id -ne $taskPlan.plan_id -or
    (@($taskPlan.methods) -join ',') -ne 'adaptive,oracle,lasso,glasso' -or
    $taskPlan.profile -ne 'full' -or $taskPlan.n_reps -ne 50) {
    throw 'Downloaded summaries are incomplete or have a mismatched plan; retain directory for inspection'
}
$taskRows = @(Import-Csv -LiteralPath (Join-Path $taskDestination 'raw_results.csv'))
if ($taskRows.Count -ne 2800 -or @($taskRows | ForEach-Object { $_.task_id + '/' + $_.method } | Sort-Object -Unique).Count -ne 2800) {
    throw 'Downloaded method rows are missing or duplicated'
}
'DOWNLOADED: 700/2800 merged summary identity checked; remote model audits remain in original output'
