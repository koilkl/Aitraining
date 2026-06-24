# Windows Build Guide - TFLite Training

## Two Build Options

### Option 1: Desktop Window (Recommended - macOS Style)
This opens a native desktop window like on macOS.

**Build:**
```powershell
.\build_pywebview.ps1
```

**Output:** `dist\TFLiteTraining\TFLiteTraining.exe`

**Pros:**
- Native desktop app look and feel
- No browser window needed
- Better integration with Windows

**Cons:**
- May have compatibility issues on some Windows systems
- Uses pywebview (Chromium-based)

---

### Option 2: Browser Window (Fallback)
This opens your default web browser instead of a native window.

**Build:**
```powershell
.\build_final.ps1
```

**Output:** `dist\TFLiteTraining.exe`

**Pros:**
- Works on all Windows systems
- Uses standard browser (Chrome, Edge, Firefox, etc.)
- Better webcam/serial port compatibility

**Cons:**
- Opens browser instead of native window
- May have browser-specific quirks

---

## Troubleshooting

### Option 1 fails (recursion errors, window won't open)
→ Switch to **Option 2** (browser version)

### Webcam not working
1. Make sure no other app is using the camera
2. Try the **Browser version** (Option 2)
3. Check Windows camera permissions in Settings

### Serial port issues
→ Use the **Browser version** (Option 2) for better hardware access

---

## Quick Start

1. Choose your version above
2. Run the build script
3. Wait 10-20 minutes for build
4. Run the .exe from `dist` folder
5. If issues, try the other version

---

## 📦 Distributing to Others (Sharing with Friends)

### Step 1: Build the EXE
```powershell
.\build_pywebview.ps1
# or
.\build_final.ps1
```

### Step 2: Create Distribution Package
```powershell
.\create_distribution.ps1
```

This will:
- Automatically detect which version you built
- Create a folder in `dist-packages\`
- Add a README.txt with instructions
- Create a ZIP file ready to share

### Step 3: Share the Files
Upload to:
- Google Drive / OneDrive / Dropbox
- Email (if file is small enough)
- USB drive
- Any file sharing service

### Recipient Instructions
1. Download the ZIP file
2. Unzip it
3. Double-click `TFLiteTraining.exe`
4. Done!

---

## Files

- `TFLiteTraining_pywebview.spec` - Desktop window version spec
- `TFLiteTraining_final.spec` - Browser window version spec
- `build_pywebview.ps1` - Build script for desktop version
- `build_final.ps1` - Build script for browser version
- `create_distribution.ps1` - Create shareable package
- `desktop_launcher_fixed.py` - Entry point for desktop version
- `desktop_launcher_simple.py` - Entry point for browser version
