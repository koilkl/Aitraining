# TFLiteTraining Windows Build Script
# Usage: .\build_windows.ps1 [-Clean] [-SkipInstall] [-OutputDir <path>]
param(
    [switch]$Clean,
    [switch]$SkipInstall,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== TFLiteTraining Windows Build ===" -ForegroundColor Cyan

# 1. Python version check — the app is built/tested against Python 3.11
#    (pywebview + pythonnet are known-good there). Prefer `py -3.11`.
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
$PY = $null
py -3.11 --version *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  $(py -3.11 --version) (using py -3.11)" -ForegroundColor Green
    $PY = "py -3.11"
} else {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Python not found. Install Python 3.11 from https://python.org" -ForegroundColor Red
        exit 1
    }
    Write-Host "  $pythonVersion (falling back to system python)" -ForegroundColor Yellow
    $PY = "python"
}

# 2. Virtual environment
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $ScriptDir ".venv"
if ($Clean -and (Test-Path $venvPath)) {
    Write-Host "  Removing existing .venv..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $venvPath
}
if (-not (Test-Path $venvPath)) {
    Invoke-Expression "$PY -m venv $venvPath"
    Write-Host "  Created .venv" -ForegroundColor Green
} else {
    Write-Host "  Using existing .venv" -ForegroundColor Green
}

# 3. Activate and install dependencies
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
$activateScript = Join-Path (Join-Path $venvPath "Scripts") "Activate.ps1"
. $activateScript

if (-not $SkipInstall) {
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m pip install pyinstaller
    Write-Host "  Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  Skipped (--SkipInstall)" -ForegroundColor Gray
}

# 4. Clean previous build
Write-Host "[4/5] Building with PyInstaller..." -ForegroundColor Yellow
$distDir = Join-Path $ScriptDir "dist"
$buildDir = Join-Path $ScriptDir "build"

# Close any running instance first, otherwise its .exe is locked and the
# rebuild fails with "PermissionError: [WinError 5] 拒绝访问".
$running = Get-Process -Name TFLiteTraining -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "  Closing running TFLiteTraining.exe..." -ForegroundColor Yellow
    $running | Stop-Process -Force
    Start-Sleep -Milliseconds 800
}

# Also reap leaked WebView2 subprocesses (msedgewebview2.exe) that can keep
# dist locked via their inherited working directory.
$leaked = Get-Process -Name msedgewebview2 -ErrorAction SilentlyContinue
if ($leaked) {
    Write-Host "  Reaping leaked WebView2 subprocesses..." -ForegroundColor Yellow
    $leaked | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

if ($Clean) {
    foreach ($dir in @($distDir, $buildDir)) {
        if (-not (Test-Path $dir)) { continue }
        $removed = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            if (-not (Test-Path $dir)) { $removed = $true; break }
            Start-Sleep -Seconds 1
        }
        if (-not $removed) {
            Write-Host "ERROR: Cannot remove $dir - another program is using it." -ForegroundColor Red
            Write-Host "  Most common cause: a terminal whose current directory is inside dist" -ForegroundColor Yellow
            Write-Host "  (e.g. a VS Code terminal tab where you ran 'cd dist'). Fix: cd .. in" -ForegroundColor Yellow
            Write-Host "  that tab, or close the tab. Also check for a File Explorer window open" -ForegroundColor Yellow
            Write-Host "  inside dist, or a still-running app." -ForegroundColor Yellow
            exit 1
        }
    }
}

python -m PyInstaller --clean --noconfirm TFLiteTraining.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyInstaller build failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Build complete" -ForegroundColor Green

# 5. Verify output
Write-Host "[5/5] Verifying output..." -ForegroundColor Yellow
$exePath = Join-Path (Join-Path $distDir "TFLiteTraining") "TFLiteTraining.exe"
if (Test-Path $exePath) {
    $exeInfo = Get-Item $exePath
    $sizeMB = [math]::Round($exeInfo.Length / 1MB, 1)
    Write-Host "  Output: $exePath" -ForegroundColor Green
    Write-Host "  Size: ${sizeMB} MB" -ForegroundColor Green
} else {
    Write-Host "ERROR: TFLiteTraining.exe not found in dist\TFLiteTraining\" -ForegroundColor Red
    Write-Host "  Contents of dist\:" -ForegroundColor Gray
    Get-ChildItem -Recurse (Join-Path $distDir "TFLiteTraining") | ForEach-Object { Write-Host "    $($_.FullName)" }
    exit 1
}

# Optionally copy to custom output dir
if ($OutputDir) {
    $outPath = [System.IO.Path]::GetFullPath($OutputDir)
    Write-Host "  Copying to $outPath ..." -ForegroundColor Gray
    New-Item -ItemType Directory -Force -Path $outPath | Out-Null
    Copy-Item -Recurse -Force (Join-Path $distDir "TFLiteTraining") $outPath
    Write-Host "  Copied to $outPath" -ForegroundColor Green
}

Write-Host "=== Build Succeeded ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launch: $exePath" -ForegroundColor White
Write-Host ""
Write-Host "Note: If you see missing DLL errors, install:" -ForegroundColor Gray
Write-Host "  Microsoft Visual C++ Redistributable 2015-2022 (x64)" -ForegroundColor Gray
Write-Host "  https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Gray
