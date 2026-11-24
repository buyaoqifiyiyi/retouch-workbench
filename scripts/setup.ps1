<#
.SYNOPSIS
    One-command fresh-PC setup for Retouch Workbench (%LOCALAPPDATA%\RetouchTools).
.EXAMPLE
    .\scripts\setup.ps1 -Tier core -DryRun
    .\scripts\setup.ps1 -Tier full -SyncSkill
#>
[CmdletBinding()]
param(
    [ValidateSet('core', 'full')]
    [string]$Tier = 'core',
    [switch]$DryRun,
    [switch]$SyncSkill,
    [string]$ToolsRoot = (Join-Path $env:LOCALAPPDATA 'RetouchTools')
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$CoreVenv = Join-Path $ToolsRoot 'venv'
$CorePy = Join-Path $CoreVenv 'Scripts\python.exe'
$GenVenv = Join-Path $ToolsRoot 'venv-gen'
$GenPy = Join-Path $GenVenv 'Scripts\python.exe'

function Find-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $ver = & py -3.12 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver.Trim() -eq '3.12') {
                return @('py', '-3.12')
            }
        } catch {}
    }
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $ver = & $pyCmd.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver.Trim() -eq '3.12') {
            return @($pyCmd.Source)
        }
    }
    throw 'Python 3.12 is required (>=3.12,<3.13). Install Python 3.12 first.'
}

$basePy = Find-Python312
Write-Host "[Retouch Setup] Repo: $RepoRoot | Tier: $Tier | ToolsRoot: $ToolsRoot"

if (-not (Test-Path -LiteralPath $CorePy)) {
    if ($DryRun) {
        Write-Host "[DryRun] Would create core venv: $($basePy -join ' ') -m venv `"$CoreVenv`""
    } else {
        New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
        if ($basePy.Count -gt 1) {
            & $basePy[0] $basePy[1] -m venv $CoreVenv
        } else {
            & $basePy[0] -m venv $CoreVenv
        }
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$reqCore = Join-Path $RepoRoot 'scripts\requirements-ai.txt'
$dlScript = Join-Path $RepoRoot 'scripts\download_models.py'

if ($DryRun) {
    Write-Host "[DryRun] Would install core dependencies: $CorePy -m pip install -r `"$reqCore`""
    Write-Host "[DryRun] Would download core models + Real-ESRGAN: python `"$dlScript`" core"
} else {
    & $CorePy -m pip install --upgrade pip
    & $CorePy -m pip install -r $reqCore
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $CorePy $dlScript core
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Tier -eq 'full') {
    $reqGen = Join-Path $RepoRoot 'scripts\requirements-gen.txt'
    $wheelsDir = Join-Path $ToolsRoot 'wheels'
    if ($DryRun) {
        Write-Host "[DryRun] Would download CUDA PyTorch wheels + generative weights: python `"$dlScript`" wheels && gen"
        Write-Host "[DryRun] Would create isolated gen venv: $GenVenv and install torch CUDA 12.4 + `"$reqGen`""
    } else {
        & $CorePy $dlScript wheels
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not (Test-Path -LiteralPath $GenPy)) {
            if ($basePy.Count -gt 1) {
                & $basePy[0] $basePy[1] -m venv $GenVenv
            } else {
                & $basePy[0] -m venv $GenVenv
            }
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        $whls = Get-ChildItem -Path $wheelsDir -Filter '*.whl' | Select-Object -ExpandProperty FullName
        & $GenPy -m pip install --upgrade pip
        & $GenPy -m pip install @whls
        & $GenPy -m pip install -r $reqGen
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $CorePy $dlScript gen
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

if ($SyncSkill) {
    $dstSkill = Join-Path $env:USERPROFILE '.agents\skills\retouch-workbench'
    if ($DryRun) {
        Write-Host "[DryRun] Would link or sync retouch-workbench skill to '$dstSkill'"
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $dstSkill -Parent) | Out-Null
        if (-not (Test-Path -LiteralPath $dstSkill)) {
            cmd /c mklink /J "`"$dstSkill`"" "`"$RepoRoot`"" | Out-Null
        }
        Write-Host "[Retouch Setup] Skill available at $dstSkill"
    }
}

$pcExe = Join-Path $env:LOCALAPPDATA 'Programs\ArtCraft\PhotoCraft\photocraft-cli.exe'
$lcExe = Join-Path $env:LOCALAPPDATA 'Programs\ArtCraft\LightCraft\lightcraft-cli.exe'
if (-not (Test-Path -LiteralPath $pcExe)) {
    Write-Host '[Notice] PhotoCraft CLI not found. Download from https://github.com/storytold/photocraft/releases/latest'
}
if (-not (Test-Path -LiteralPath $lcExe)) {
    Write-Host '[Notice] LightCraft CLI not found. Download from https://github.com/storytold/lightcraft/releases/latest'
}

if (-not $DryRun -and (Test-Path -LiteralPath $CorePy)) {
    & $CorePy (Join-Path $RepoRoot 'scripts\retouch-ai.py') doctor
}
