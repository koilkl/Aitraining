# Debug Build Script
Write-Host "=== TFLiteTraining DEBUG Build Tool ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Virtual environment found, activating..." -ForegroundColor Yellow
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
Write-Host "Starting debug build..." -ForegroundColor Green
Write-Host "This will take several minutes..." -ForegroundColor Yellow
pyinstaller TFLiteTraining_debug.spec

# Check build result
if (Test-Path "dist\TFLiteTraining_Debug.exe") {
    Write-Host ""
    Write-Host "🎉 DEBUG Build successful!" -ForegroundColor Green
    Write-Host "Debug EXE location: $(Get-Location)\dist\TFLiteTraining_Debug.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT: Run this EXE - it will keep console open even if it crashes!" -ForegroundColor Yellow
    Write-Host "You will see the error messages clearly now!" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Build may have failed, please check errors above." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
