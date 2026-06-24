#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to see what errors without pywebview
"""

print("=== Simple Test ===")
print()

print("1. Testing imports...")
try:
    import sys
    print(f"   Python: {sys.version}")
    import os
    print("   os imported")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Testing streamlit...")
    import streamlit
    print(f"   Streamlit: {streamlit.__version__}")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing pywebview...")
    import webview
    print(f"   pywebview available")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4. Testing main imports...")
    import camera_permission
    import dataset_io
    import trainer
    print("   All imports successful!")
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing complete!")
print("Now trying to run the real app...")
print()

try:
    print("5. Starting desktop_launcher...")
    exec(open("desktop_launcher.py", encoding='utf-8').read())
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nPress Enter to exit...")
input()
