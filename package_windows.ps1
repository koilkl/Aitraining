# TFLiteTraining Windows Packaging Script
# Packages dist\TFLiteTraining into a distributable .zip (the Windows
# equivalent of the macOS .dmg).
#
# Usage:
#   .\package_windows.ps1
#   .\package_windows.ps1 -Version 2.4.14 -OutputDir C:\releases
param(
    [string]$Version = "2.4.14",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== TFLiteTraining Windows Packaging ===" -ForegroundColor Cyan

$distDir = Join-Path $ScriptDir "dist"
$appDir = Join-Path $distDir "TFLiteTraining"
$exePath = Join-Path $appDir "TFLiteTraining.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "ERROR: $exePath not found." -ForegroundColor Red
    Write-Host "       Run .\build_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$zipName = "TFLiteTraining-windows-x64-v$Version.zip"
$zipPath = Join-Path $distDir $zipName

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Write-Host "  Zipping $appDir" -ForegroundColor Gray
Write-Host "  -> $zipPath" -ForegroundColor Gray
Write-Host "  (a ~1 GB bundle can take a few minutes)" -ForegroundColor Gray

# Prefer tar (fast, built into Windows 10+), fall back to Compress-Archive.
$tar = Get-Command tar -ErrorAction SilentlyContinue
if ($tar) {
    & tar -a -c -f $zipPath -C $distDir "TFLiteTraining"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: tar failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Compress-Archive -Path $appDir -DestinationPath $zipPath -Force
}

$zipInfo = Get-Item $zipPath
$sizeMB = [math]::Round($zipInfo.Length / 1MB, 1)
Write-Host "  Output: $zipPath" -ForegroundColor Green
Write-Host "  Size:   ${sizeMB} MB" -ForegroundColor Green

if ($OutputDir) {
    $outPath = [System.IO.Path]::GetFullPath($OutputDir)
    New-Item -ItemType Directory -Force -Path $outPath | Out-Null
    Copy-Item -Force $zipPath $outPath
    Write-Host "  Copied to $outPath" -ForegroundColor Green
}

Write-Host "=== Packaging Succeeded ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Distribute: $zipName" -ForegroundColor White
Write-Host "Recipients extract the zip and run TFLiteTraining\TFLiteTraining.exe." -ForegroundColor Gray
