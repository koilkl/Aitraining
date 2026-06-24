# TFLite Training - Final Windows Build Script
Write-Host "========================================" -ForegroundColor Green
Write-Host "  TFLite Training - Windows Build Tool" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
}

# Clean old build files
Write-Host "Cleaning old build files..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force build
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force dist
}

# Build
Write-Host ""
Write-Host "Building application..." -ForegroundColor Green
Write-Host "This will take several minutes, please be patient..." -ForegroundColor Yellow
Write-Host ""
.\.venv\Scripts\pyinstaller.exe TFLiteTraining_final.spec

# Check build result
Write-Host ""
if (Test-Path "dist\TFLiteTraining.exe") {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Build SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Application is at: $(Get-Location)\dist\TFLiteTraining.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "How to use:" -ForegroundColor Yellow
    Write-Host "1. Double-click TFLiteTraining.exe to run"
    Write-Host "2. An app window will open using Edge/Chrome app mode"
    Write-Host "3. If no supported browser is found, it will fall back to the default browser"
    Write-Host "4. Close the app window when you're done"
    Write-Host ""
    Write-Host "Note: This version avoids pywebview and uses a more stable browser-engine app window on Windows."
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Build FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above"
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
