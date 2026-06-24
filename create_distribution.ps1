# TFLite Training - Distribution Package
# Run this after building to create a distributable ZIP file

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Creating Distribution Package" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check which build exists
$desktopBuild = "dist\TFLiteTraining\TFLiteTraining.exe"
$browserBuild = "dist\TFLiteTraining.exe"

$buildPath = ""
$packageName = ""

if (Test-Path $desktopBuild) {
    Write-Host "Found Desktop Window version" -ForegroundColor Green
    $buildPath = "dist\TFLiteTraining"
    $packageName = "TFLiteTraining-Desktop-Windows"
} elseif (Test-Path $browserBuild) {
    Write-Host "Found Browser version" -ForegroundColor Green
    $buildPath = "dist\TFLiteTraining.exe"
    $packageName = "TFLiteTraining-Browser-Windows"
} else {
    Write-Host "ERROR: No build found!" -ForegroundColor Red
    Write-Host "Please run build_pywebview.ps1 or build_final.ps1 first" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Create output directory
$outputDir = "dist-packages"
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

# Copy build to distribution folder
$distFolder = "$outputDir\$packageName"
if (Test-Path $distFolder) {
    Remove-Item -Recurse -Force $distFolder
}
New-Item -ItemType Directory -Force -Path $distFolder | Out-Null

if ($buildPath -eq "dist\TFLiteTraining") {
    # Desktop version - copy entire folder
    Copy-Item -Recurse $buildPath -Destination $distFolder
} else {
    # Browser version - copy single file
    Copy-Item $buildPath -Destination $distFolder
}

# Create README for recipients
$readmeContent = @"
# TFLite Training

Desktop AI training tool for creating image classification models.

## How to Use

1. Double-click `TFLiteTraining.exe` to start
2. Wait for the window to open (may take a few seconds)
3. Start creating your AI model!

## System Requirements

- Windows 10 or Windows 11
- Internet connection (for first run to install dependencies)
- Webcam (optional, for capturing images)
- USB cable (optional, for connecting devices)

## Tips

- If window doesn't open, try running as administrator
- Close other apps using the webcam before starting
- Your projects are saved in: %APPDATA%\TFLiteTraining

## Troubleshooting

**Window won't open?**
- Try running as administrator
- Make sure WebView2 Runtime is installed

**Webcam not working?**
- Close other apps using the webcam
- Check Windows camera permissions

**App runs slow?**
- Close other programs
- Make sure antivirus isn't scanning the folder

---
Made with ❤️ for AI education
"@

$readmeContent | Out-File -FilePath "$distFolder\README.txt" -Encoding UTF8

# Create ZIP file
$zipPath = "dist-packages\$packageName.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath
}

Write-Host "Creating ZIP package..." -ForegroundColor Yellow
Compress-Archive -Path $distFolder -DestinationPath $zipPath -CompressionLevel Optimal

# Show results
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Package Created!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Files ready to share:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Folder (uncompressed):" -ForegroundColor White
Write-Host "   $distFolder" -ForegroundColor Gray
Write-Host ""
Write-Host "2. ZIP file (for sharing):" -ForegroundColor White
Write-Host "   $zipPath" -ForegroundColor Gray
Write-Host ""

$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host "ZIP size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor White
Write-Host ""

Write-Host "To share with others:" -ForegroundColor Yellow
Write-Host "1. Upload $zipPath to Google Drive, OneDrive, etc." -ForegroundColor White
Write-Host "2. Or email the file directly" -ForegroundColor White
Write-Host "3. Recipients just unzip and run!" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
