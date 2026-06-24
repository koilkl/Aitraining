#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug version - keeps console open
"""
from __future__ import annotations

import sys
import os

print("=" * 60)
print("TFLiteTraining Launcher - DEBUG VERSION")
print("=" * 60)
print()

# Print Python info
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Executable: {sys.executable}")
print(f"Working dir: {os.getcwd()}")
print()

# Add the current directory to path
if hasattr(sys, '_MEIPASS'):
    print(f"Running in PyInstaller mode, _MEIPASS={sys._MEIPASS}")
    # In PyInstaller, we need to add the extracted dir to path
    sys.path.insert(0, sys._MEIPASS)
else:
    print("Running normally, not in PyInstaller mode")

print()
print("-" * 60)
print("Importing modules...")
print("-" * 60)

try:
    import contextlib
    import json
    import multiprocessing
    import socket
    import threading
    import time
    import urllib.request
    from pathlib import Path

    print("✓ Basic imports OK")

    # Import webview first
    print("Importing webview...")
    import webview
    print("✓ webview imported")

except Exception as e:
    print()
    print("=" * 60)
    print("ERROR IN INITIAL IMPORT")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Press Enter to exit...")
    input()
    sys.exit(1)

print()
print("-" * 60)
print("Starting main app...")
print("-" * 60)

try:
    # Now import the real launcher
    from desktop_launcher import main
    print("Calling main()...")
    main()
    print("main() exited normally")

except Exception as e:
    print()
    print("=" * 60)
    print("CRASHED!")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Program finished. Press Enter to exit...")
print("=" * 60)
input()
