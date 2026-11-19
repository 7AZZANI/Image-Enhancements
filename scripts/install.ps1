# AZZANI Image Enhancements -- Windows 1-Line Auto-GPU Installer (MIT License)
# Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/7AZZANI/Image-Enhancements/main/scripts/install.ps1 | iex"
#   .\scripts\install.ps1 -Extras "directml,raw"

param(
    [string]$InstallDir = $(Join-Path $env:LOCALAPPDATA "azzani-enhance"),
    [string]$RepoUrl = "https://github.com/7AZZANI/Image-Enhancements.git",
    [string]$Extras = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @("py", "-3") }
        } catch {}
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return @("python") }
        } catch {}
    }
    return $null
}

$LocalRoot = $null
if ($PSScriptRoot) {
    $Cand = Resolve-Path (Join-Path $PSScriptRoot "..") -ErrorAction SilentlyContinue
    if ($Cand -and (Test-Path (Join-Path $Cand "pyproject.toml")) -and (Test-Path (Join-Path $Cand "enhance.py"))) {
        $LocalRoot = $Cand.Path
    }
}
if ($LocalRoot) { $InstallDir = $LocalRoot }

if ($DryRun) {
    Write-Host "[azzani-install] Dry-run plan:"
    Write-Host "  InstallDir : $InstallDir"
    Write-Host "  Extras     : $Extras"
    exit 0
}

$PyCmd = Resolve-PythonCommand
if (-not $PyCmd) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "[azzani-install] Python 3.10+ not found; installing standalone uv..."
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }
    if (-not $LocalRoot) {
        New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir -Parent) | Out-Null
        if (Test-Path (Join-Path $InstallDir ".git")) {
            git -C $InstallDir pull --ff-only
        } else {
            git clone --depth 1 $RepoUrl $InstallDir
        }
    }
    Set-Location $InstallDir
    if (-not (Test-Path (Join-Path $InstallDir ".venv"))) {
        & uv venv --python 3.12 (Join-Path $InstallDir ".venv")
    }
    $PyExe = Join-Path $InstallDir ".venv\Scripts\python.exe"
} else {
    if (-not $LocalRoot) {
        New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir -Parent) | Out-Null
        if (Test-Path (Join-Path $InstallDir ".git")) {
            git -C $InstallDir pull --ff-only
        } else {
            git clone --depth 1 $RepoUrl $InstallDir
        }
    }
    Set-Location $InstallDir
    $PyExe = $PyCmd[0]
    $PyArgs = @()
    if ($PyCmd.Count -gt 1) { $PyArgs = $PyCmd[1..($PyCmd.Count - 1)] }
    $env:AZZANI_AUTO_SETUP = "1"
    & $PyExe @PyArgs (Join-Path $InstallDir "enhance.py") --setup --yes
    $VenvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"
    if (Test-Path $VenvPy) { $PyExe = $VenvPy }
}

if ($PyExe -like "*\.venv\*") {
    $env:AZZANI_AUTO_SETUP = "1"
    & $PyExe (Join-Path $InstallDir "enhance.py") --setup --yes
}

if ($Extras.Trim()) {
    Write-Host "[azzani-install] Installing optional extras: [$Extras]"
    & (Join-Path $InstallDir ".venv\Scripts\python.exe") -m pip install -e "${InstallDir}[${Extras}]"
}

$ShimDir = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
if (Test-Path $ShimDir) {
    $ShimPath = Join-Path $ShimDir "azzani-enhance.cmd"
    $CmdBody = "@echo off`r`n`"$InstallDir\.venv\Scripts\python.exe`" `"$InstallDir\enhance.py`" %*`r`n"
    Set-Content -Path $ShimPath -Value $CmdBody -Encoding ASCII
    Write-Host "[azzani-install] CLI launcher ready: azzani-enhance ($ShimPath)"
}

& (Join-Path $InstallDir ".venv\Scripts\python.exe") (Join-Path $InstallDir "enhance.py") --doctor
