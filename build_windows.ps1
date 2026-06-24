# Windows Build Script

Write-Host "=== TFLiteTraining Windows Build Tool ===" -ForegroundColor Green

# Check if virtual environment exists
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

# Check if PyInstaller is installed
Write-Host "Checking PyInstaller..." -ForegroundColor Yellow
try {
    pyinstaller --version | Out-Null
} catch {
    Write-Host "PyInstaller not found, installing..." -ForegroundColor Red
    pip install pyinstaller
}

# Start building
Write-Host "Starting application build with PyInstaller..." -ForegroundColor Green
Write-Host "This will take several minutes, please wait..." -ForegroundColor Yellow
pyinstaller TFLiteTraining.spec

# Check build result
if (Test-Path "dist\TFLiteTraining.exe") {
    Write-Host ""
    Write-Host "🎉 Build successful!" -ForegroundColor Green
    Write-Host "Application location: $(Get-Location)\dist\TFLiteTraining.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT: Console is enabled for debugging!" -ForegroundColor Yellow
    Write-Host "Run the EXE and you will see a console window with debug info." -ForegroundColor Yellow
    Write-Host ""
} elseif (Test-Path "dist\TFLiteTraining\TFLiteTraining.exe") {
    Write-Host ""
    Write-Host "🎉 Build successful!" -ForegroundColor Green
    Write-Host "Application location: $(Get-Location)\dist\TFLiteTraining\TFLiteTraining.exe" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Build may have failed, please check the error messages above." -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
