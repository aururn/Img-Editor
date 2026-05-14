param(
    [Parameter(Mandatory = $true)]
    [Alias("Host", "IP")]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [Alias("PortNumber")]
    [int]$Port,

    [string]$User = "root",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_vast",
    [string]$RemoteWorkspace = "/workspace",
    [string]$RemoteAppDir = "/workspace/ImgEditor",
    [string]$ZipPath = "$env:USERPROFILE\Documents\ImgEditor_code.zip",
    [string]$CivitaiToken = "",
    [switch]$SkipCheckpoints,
    [switch]$NoInstall,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Convert-SecureStringToPlainText {
    param([securestring]$SecureString)
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $Name"
    }
}

Require-Command scp
Require-Command ssh

if ($HostName -match "^(?<remoteUser>[^@]+)@(?<remoteHost>.+)$") {
    if ($PSBoundParameters.ContainsKey("User") -and $User -ne $Matches.remoteUser) {
        throw "HostName contains user '$($Matches.remoteUser)' but -User is '$User'. Use either -HostName $($Matches.remoteHost) -User $User or -HostName $HostName."
    }
    $User = $Matches.remoteUser
    $HostName = $Matches.remoteHost
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$zipParent = Split-Path -Parent $ZipPath
if ($zipParent -and -not (Test-Path $zipParent)) {
    New-Item -ItemType Directory -Path $zipParent | Out-Null
}

if (-not $SkipCheckpoints -and [string]::IsNullOrWhiteSpace($CivitaiToken)) {
    $secure = Read-Host "Civitai API token (checkpoint download; empty is not recommended)" -AsSecureString
    $CivitaiToken = Convert-SecureStringToPlainText $secure
}

Write-Host "[1/4] Create zip: $ZipPath"
$include = @(
    "app.py",
    "config.yaml",
    "requirements.txt",
    "src",
    "presets",
    "README.md",
    "docs",
    "scripts"
) | ForEach-Object {
    Join-Path $repoRoot $_
}
Compress-Archive -Path $include -DestinationPath $ZipPath -Force

$remote = "$User@$HostName"
$remoteZip = "$RemoteWorkspace/ImgEditor_code.zip"

Write-Host "[2/4] Upload zip to ${remote}:${remoteZip}"
& scp -i $KeyPath -P $Port $ZipPath "${remote}:${remoteZip}"
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

$bootstrapArgs = @()
if ($SkipCheckpoints) {
    $bootstrapArgs += "--skip-checkpoints"
}
else {
    $bootstrapArgs += "--download-checkpoints"
}
if ($NoInstall) {
    $bootstrapArgs += "--no-install"
}
if ($Start) {
    $bootstrapArgs += "--start"
}

$tokenPrefix = ""
if (-not $SkipCheckpoints -and -not [string]::IsNullOrWhiteSpace($CivitaiToken)) {
    $tokenBytes = [Text.Encoding]::UTF8.GetBytes($CivitaiToken)
    $tokenB64 = [Convert]::ToBase64String($tokenBytes)
    $tokenPrefix = "export CIVITAI_TOKEN=`$(printf '%s' '$tokenB64' | base64 -d); "
}

$bootstrapArgText = $bootstrapArgs -join " "
$remoteCommand = @"
set -e &&
echo '[remote] enter workspace' &&
cd $RemoteWorkspace &&
echo '[remote] create app dir' &&
mkdir -p $RemoteAppDir &&
echo '[remote] extract zip' &&
(command -v unzip >/dev/null 2>&1 && unzip -o $remoteZip -d $RemoteAppDir || python3 -m zipfile -e $remoteZip $RemoteAppDir) &&
echo '[remote] enter app dir' &&
cd $RemoteAppDir &&
echo '[remote] chmod bootstrap' &&
chmod +x scripts/vast_bootstrap.sh &&
echo '[remote] run bootstrap' &&
${tokenPrefix}bash scripts/vast_bootstrap.sh $bootstrapArgText
"@ -replace "`r?`n", " "

Write-Host "[3/4] Run remote bootstrap"
& ssh -i $KeyPath -p $Port $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "ssh/bootstrap failed with exit code $LASTEXITCODE"
}

Write-Host "[4/4] Done"
if (-not $Start) {
    Write-Host ""
    Write-Host "Start app on Vast:"
    Write-Host "  ssh -i `"$KeyPath`" -p $Port $remote `"cd $RemoteAppDir && python3 app.py`""
}
Write-Host ""
Write-Host "Open tunnel from another PowerShell:"
Write-Host "  ssh -i `"$KeyPath`" -p $Port -N -L 7860:127.0.0.1:7860 $remote"
Write-Host "Then open: http://localhost:7860"
