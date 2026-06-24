# Windows Build Guide

## Problem Analysis
The previously generated exe didn't work on Windows mainly because:
1. PyInstaller config had Mac-specific BUNDLE configuration
2. Missing some necessary module references
3. Didn't include all project source files

## Solution

### Step 1: Install Dependencies
```powershell
# Activate virtual environment (if you have one)
.\.venv\Scripts\Activate.ps1

# Make sure PyInstaller is installed
pip install pyinstaller
```

### Step 2: Clean Old Build Files
```powershell
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
```

### Step 3: Build with Fixed Config
```powershell
# Use the fixed spec file
pyinstaller TFLiteTraining.spec

# Alternatively, use command line (not recommended as config is complex)
# pyinstaller --onefile --windowed --name TFLiteTraining desktop_launcher.py
```

### Step 4: Test the Built Application
```powershell
# After building, app is at dist\TFLiteTraining\TFLiteTraining.exe
# Or run directly:
.\dist\TFLiteTraining\TFLiteTraining.exe
```

## Debug Tips

### Enable Console Output (for Debugging)
If you need to see error logs, temporarily modify `TFLiteTraining.spec` and change `console=False` to `console=True`.

### View Log Files
The app creates log files at:
```
%APPDATA%\TFLiteTraining\logs\streamlit.log
```

## Usage Instructions

1. **Launch App**: Double-click `dist\TFLiteTraining\TFLiteTraining.exe`
2. **Import Images**: Can use ZIP files, multiple images, or local folders
3. **Add Samples**: Can capture images from camera or device stream
4. **Train Model**: Configure parameters and train
5. **Export Model**: Export as TFLite format and C source code

## Common Issues

### 1. App Quits Immediately After Launch
- Check console output (if console=True enabled)
- Check log file at %APPDATA%\TFLiteTraining\logs\streamlit.log

### 2. Camera Can't Open
- Ensure no other apps are using the camera
- Check camera permissions in Windows privacy settings

### 3. Missing Module Errors
- Make sure all dependencies are installed in the build environment
- Check hiddenimports in TFLiteTraining.spec includes required modules

### 4. Build Errors
- Ensure Python version and dependencies are compatible
- Try disabling UPX compression (change upx=True to upx=False in spec file)

## Project Source Files
Make sure all following files are included correctly:
- app.py
- camera_permission.py
- dataset_io.py
- trainer.py
- ui_styles.py
- serial_device.py
- record_controller.py
- device_stream.py
- tflite_train.py
- desktop_launcher.py
