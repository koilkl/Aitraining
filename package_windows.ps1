# TFLiteTraining Windows Packaging Script
# Builds a distributable installer .exe (Inno Setup) from dist\TFLiteTraining,
# falling back to a .zip when Inno Setup is not installed.
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

# --- Locate Inno Setup compiler (ISCC.exe) ---
$iscc = $null
$candidatePaths = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)
foreach ($p in $candidatePaths) {
    if ($p -and (Test-Path $p)) { $iscc = $p; break }
}
if (-not $iscc) {
    $isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($isccCmd) { $iscc = $isccCmd.Source }
}

if ($iscc) {
    # --- Build a real installer .exe ---
    Write-Host "  Inno Setup found: $iscc" -ForegroundColor Green
    Write-Host "  Building installer (a ~1 GB bundle can take a few minutes)..." -ForegroundColor Gray
    & $iscc /DAppVersion=$Version "installer.iss"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Inno Setup failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    $installerPath = Join-Path $distDir "TFLiteTraining-Setup-v$Version.exe"
    if (-not (Test-Path $installerPath)) {
        Write-Host "ERROR: installer was not produced at $installerPath" -ForegroundColor Red
        exit 1
    }
    $info = Get-Item $installerPath
    Write-Host "  Output: $installerPath" -ForegroundColor Green
    Write-Host "  Size:   $([math]::Round($info.Length / 1MB, 1)) MB" -ForegroundColor Green

    if ($OutputDir) {
        $outPath = [System.IO.Path]::GetFullPath($OutputDir)
        New-Item -ItemType Directory -Force -Path $outPath | Out-Null
        Copy-Item -Force $installerPath $outPath
        Write-Host "  Copied to $outPath" -ForegroundColor Green
    }
    Write-Host "=== Packaging Succeeded (installer) ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Distribute: TFLiteTraining-Setup-v$Version.exe" -ForegroundColor White
} else {
    # --- Inno Setup not installed: fall back to .zip ---
    Write-Host "  WARNING: Inno Setup not found." -ForegroundColor Yellow
    Write-Host "    Install it for a real installer .exe:" -ForegroundColor Yellow
    Write-Host "      winget install JRSoftware.InnoSetup" -ForegroundColor Yellow
    Write-Host "      (or download from https://jrsoftware.org/isinfo.php)" -ForegroundColor Yellow
    Write-Host "  Falling back to a .zip..." -ForegroundColor Gray

    $zipName = "TFLiteTraining-windows-x64-v$Version.zip"
    $zipPath = Join-Path $distDir $zipName
    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

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

    $info = Get-Item $zipPath
    Write-Host "  Output: $zipPath" -ForegroundColor Green
    Write-Host "  Size:   $([math]::Round($info.Length / 1MB, 1)) MB" -ForegroundColor Green

    if ($OutputDir) {
        $outPath = [System.IO.Path]::GetFullPath($OutputDir)
        New-Item -ItemType Directory -Force -Path $outPath | Out-Null
        Copy-Item -Force $zipPath $outPath
        Write-Host "  Copied to $outPath" -ForegroundColor Green
    }
    Write-Host "=== Packaging Succeeded (zip fallback) ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Distribute: $zipName" -ForegroundColor White
    Write-Host "Recipients extract the zip and run TFLiteTraining\TFLiteTraining.exe." -ForegroundColor Gray
}
