# TFLite Training - Fix Virtual Environment
# This script will recreate the virtual environment and install all dependencies

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  Fixing Virtual Environment" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

# Step 1: Remove corrupted virtual environment
Write-Host "[1/4] Removing corrupted virtual environment..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
    Write-Host "✓ Removed .venv" -ForegroundColor Green
} else {
    Write-Host "✓ No .venv found" -ForegroundColor Green
}

# Step 2: Create new virtual environment
Write-Host ""
Write-Host "[2/4] Creating new virtual environment..." -ForegroundColor Cyan
python -m venv .venv
Write-Host "✓ Created new .venv" -ForegroundColor Green

# Step 3: Activate and upgrade pip
Write-Host ""
Write-Host "[3/4] Upgrading pip..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Step 4: Install all dependencies
Write-Host ""
Write-Host "[4/4] Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Virtual Environment Fixed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Now you can build the app:" -ForegroundColor White
Write-Host "  .\build_pywebview.ps1" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
