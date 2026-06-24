"""
Quick test to see if pywebview works
"""
import sys
import os

print("="*60)
print("Testing pywebview...")
print(f"Platform: {sys.platform}")
print("="*60)

# Test 1: Can we import pywebview?
print("\n[Test 1] Importing pywebview...")
try:
    import pywebview
    print(f"✓ pywebview version: {pywebview.__version__ if hasattr(pywebview, '__version__') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to import pywebview: {e}")
    sys.exit(1)

# Test 2: Can we create a window object?
print("\n[Test 2] Creating window...")
try:
    window = pywebview.create_window(
        'Test Window',
        'https://httpbin.org/html',
        width=800,
        height=600
    )
    print("✓ Window created successfully")
except Exception as e:
    print(f"✗ Failed to create window: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Can we start the GUI? (will try for 3 seconds)
print("\n[Test 3] Starting GUI (will exit after 3 seconds)...")
print("If this works, pywebview should open a window.")
print("If it crashes or gives recursion errors, there's a system issue.")
print()

try:
    # This will block, but we set a timeout in JavaScript
    import threading
    import time
    
    def start_gui():
        try:
            pywebview.start(debug=False)
        except Exception as e:
            print(f"GUI Error: {e}")
            import traceback
            traceback.print_exc()
    
    t = threading.Thread(target=start_gui)
    t.daemon = True
    t.start()
    
    # Wait 3 seconds
    time.sleep(3)
    print("Test complete (3 second timeout)")
    
except KeyboardInterrupt:
    print("Test interrupted")
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("If you saw a window open, pywebview works!")
print("If you got recursion errors, we need a different approach.")
print("="*60)
