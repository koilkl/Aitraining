#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify main dependencies work correctly
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

print("=== TFLiteTraining Dependencies Test ===")
print(f"Python Version: {sys.version}")
print(f"OS: {sys.platform}")
print()

# Test 1: Import basic dependencies
print("1. Testing dependencies...")
deps = [
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("streamlit", "Streamlit"),
    ("pywebview", "pywebview"),
    ("serial", "PySerial"),
    ("websocket", "websocket"),
]

all_ok = True
for dep, name in deps:
    try:
        __import__(dep)
        print(f"   [OK] {name} imported successfully")
    except ImportError as e:
        print(f"   [FAIL] {name} import failed: {e}")
        all_ok = False

print()

# Test 2: Check all source files
print("2. Checking source files...")
source_files = [
    "app.py",
    "camera_permission.py",
    "dataset_io.py",
    "trainer.py",
    "ui_styles.py",
    "serial_device.py",
    "record_controller.py",
    "device_stream.py",
    "tflite_train.py",
    "desktop_launcher.py",
]

project_root = Path(__file__).parent
for fname in source_files:
    fpath = project_root / fname
    if fpath.exists():
        print(f"   [OK] {fname} exists")
    else:
        print(f"   [FAIL] {fname} missing")
        all_ok = False

print()

# Test 3: Check PyInstaller config
print("3. Checking build config...")
spec_path = project_root / "TFLiteTraining.spec"
if spec_path.exists():
    print(f"   [OK] TFLiteTraining.spec exists")
    print("   Config preview:")
    print("   ------------------")
    with open(spec_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:30], 1):
            print(f"   {i:2d}: {line.rstrip()}")
        if len(lines) > 30:
            print(f"   ... (and {len(lines)-30} more lines)")
else:
    print(f"   [FAIL] TFLiteTraining.spec missing")
    all_ok = False

print()

# Test 4: Try importing our modules
print("4. Testing project modules...")
try:
    from camera_permission import get_camera_access_status
    status = get_camera_access_status()
    print(f"   [OK] camera_permission imported")
    print(f"   Camera permission status: {status.status}")
except Exception as e:
    print(f"   [FAIL] camera_permission test failed: {e}")
    import traceback
    traceback.print_exc()

print()

if all_ok:
    print("=== All tests passed! Ready to build ===")
    print()
    print("Tips:")
    print("  - On Windows: .\\build_windows.ps1")
    print("  - Or manually: pyinstaller TFLiteTraining.spec")
else:
    print("=== Some issues found ===")
    print()
    print("Please fix the issues above before building.")
