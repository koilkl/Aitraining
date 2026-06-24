# TFLite Training - macOS Style Build (with pywebview window)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build: Desktop App (pywebview window)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
}

# Clean old build files
Write-Host "Cleaning old build files..." -ForegroundColor Yellow
# if (Test-Path "build") {
#     Remove-Item -Recurse -Force build
# }
# if (Test-Path "dist") {
#     Remove-Item -Recurse -Force dist
# }

# Build
Write-Host ""
Write-Host "Building application (this may take 10-20 minutes)..." -ForegroundColor Green
Write-Host ""
.\.venv\Scripts\pyinstaller.exe TFLiteTraining_pywebview.spec

# Check build result
Write-Host ""
if (Test-Path "dist\TFLiteTraining.exe") {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Build SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Application: dist\TFLiteTraining.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This version opens a native desktop window (like macOS) instead of browser." -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Build FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "If you see recursion errors, try the browser version instead:" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\pyinstaller.exe TFLiteTraining_final.spec" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
