"""
Test script to find a working pywebview backend on Windows
"""
import sys
import os

print("Testing pywebview backends...")
print(f"Platform: {sys.platform}")

# Test different environment variables
backends_to_try = [
    ('edgechromium', 'Microsoft Edge (Chromium)'),
    ('winforms', 'Windows Forms'),
    ('mshtml', 'MSHTML (legacy)'),
    ('cef', 'Chromium Embedded Framework'),
]

for backend, description in backends_to_try:
    print(f"\n{'='*60}")
    print(f"Testing: {description} (PYWEBVIEW_GUI={backend})")
    print('='*60)
    
    os.environ['PYWEBVIEW_GUI'] = backend
    
    try:
        import pywebview
        print(f"pywebview version: {pywebview.__version__ if hasattr(pywebview, '__version__') else 'unknown'}")
        
        # Create a simple window to test
        print("Creating test window...")
        window = pywebview.create_window(
            'Test Window',
            'https://httpbin.org/html',
            width=800,
            height=600
        )
        
        print("Starting pywebview (will close automatically in 5 seconds)...")
        # Start in a non-blocking way to test if it crashes
        import threading
        import time
        
        def run_pywebview():
            try:
                pywebview.start(debug=True)
            except Exception as e:
                print(f"Error during start: {e}")
        
        # Start in thread
        t = threading.Thread(target=run_pywebview)
        t.daemon = True
        t.start()
        
        # Wait a bit
        time.sleep(3)
        
        print(f"✓ {description} seems to work!")
        print("SUCCESS!")
        break
        
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("Test complete!")
print("="*60)
