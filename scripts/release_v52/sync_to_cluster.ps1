param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [string]$RemoteRoot = '/public/home/xjh/xmx/tensor-sparse-regression-main',
    [string]$LocalRoot = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$InventoryPath,
    [string]$RemotePython = 'python',
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
if ($SshHost -notmatch '^[A-Za-z0-9_.@:-]+$' -or $SshHost.StartsWith('-')) { throw 'Invalid SSH host' }
if (-not $RemoteRoot.StartsWith('/') -or $RemoteRoot -match '[\r\n\x00]') { throw 'RemoteRoot must be an absolute Linux path' }
$LocalRoot = (Resolve-Path -LiteralPath $LocalRoot).Path
$taskControl = Join-Path $LocalRoot 'scripts/release_v52/release_control.py'
$taskCode = (& python -B $taskControl remote-code) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'Unable to load local control code' }
$taskRootCode = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RemoteRoot))
if ($RemotePython -match '[\r\n\x00]' -or $RemotePython.StartsWith('-')) { throw 'Invalid remote Python command' }
$taskPython = "'" + $RemotePython.Replace("'", "'\''") + "'"
$taskPrefix = "import base64,sys;sys.argv=['release_control','inventory','--root',base64.b64decode('$taskRootCode').decode('utf-8')];exec(compile(base64.b64decode('$taskCode'),'<release-control>','exec'))"
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('v52-sync-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
try {
    if (-not $InventoryPath) {
        $InventoryPath = Join-Path $taskTemp 'remote_inventory.json'
        $taskInventory = & ssh $SshHost ($taskPython + ' -B -c "' + $taskPrefix + '"')
        if ($LASTEXITCODE -ne 0) { throw 'Remote inventory failed; no upload performed' }
        [IO.File]::WriteAllText($InventoryPath, ($taskInventory -join "`n"), (New-Object Text.UTF8Encoding($false)))
    }
    $taskPlan = & python -B $taskControl sync-plan --root $LocalRoot --inventory $InventoryPath
    if ($LASTEXITCODE -ne 0) { throw 'Source conflict; no upload performed' }
    'DRY_RUN: reviewed minimal update plan'
    $taskPlan
    if (-not $Apply) { return }
    $taskId = [Guid]::NewGuid().ToString('N')
    $taskArchive = Join-Path $taskTemp 'upload.tar'
    & python -B $taskControl package --root $LocalRoot --inventory $InventoryPath --archive $taskArchive
    if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; no upload performed' }
    $taskSetup = "import base64,pathlib;r=pathlib.Path(base64.b64decode('$taskRootCode').decode('utf-8')).resolve(strict=True);p=r/'.v52-release-staging'/'$taskId';p.mkdir(parents=True,exist_ok=False);print(str(p))"
    $taskRemoteStage = (& ssh $SshHost ($taskPython + ' -B -c "' + $taskSetup + '"')) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw 'Remote staging failed' }
    # OpenSSH 9 SFTP receives the literal path as one native argument, without shell quotes.
    $taskRemoteArchive = $taskRemoteStage + '/upload.tar'
    & scp $taskArchive ($SshHost + ':' + $taskRemoteArchive)
    if ($LASTEXITCODE -ne 0) { throw 'Archive upload failed; production not updated' }
    $taskArchiveCode = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($taskRemoteArchive))
    $taskApplyCode = "import base64,sys;sys.argv=['release_control','apply-package','--root',base64.b64decode('$taskRootCode').decode('utf-8'),'--archive',base64.b64decode('$taskArchiveCode').decode('utf-8')];exec(compile(base64.b64decode('$taskCode'),'<release-control>','exec'))"
    & ssh $SshHost ($taskPython + ' -B -c "' + $taskApplyCode + '"')
    if ($LASTEXITCODE -ne 0) { throw 'Remote hash/backup/apply failed; inspect retained staging and backups' }
    'APPLIED: explicit files only; remote backups and staging retained for audit'
} finally {
    # Keep the local reviewed inventory/archive for audit; no recursive cleanup.
    'Local sync audit directory: ' + $taskTemp
}
