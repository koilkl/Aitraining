# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
import sys
import os

# Collect all necessary data and modules
datas = [
    ('app.py', '.'), 
    ('camera_permission.py', '.'), 
    ('dataset_io.py', '.'), 
    ('trainer.py', '.'), 
    ('ui_styles.py', '.'), 
    ('serial_device.py', '.'), 
    ('record_controller.py', '.'), 
    ('device_stream.py', '.'), 
    ('tflite_train.py', '.')
]
binaries = []
hiddenimports = []

# Collect main packages
print("Collecting streamlit...")
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

print("Collecting streamlit_drawable_canvas...")
tmp_ret = collect_all('streamlit_drawable_canvas')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

print("Collecting cv2...")
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

print("Collecting websocket...")
tmp_ret = collect_all('websocket')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

print("Collecting PIL...")
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

print("Collecting tensorflow...")
try:
    tmp_ret = collect_all('tensorflow')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    print("Warning: Could not collect tensorflow fully, will add manually")
    pass

print("Collecting pywebview...")
try:
    tmp_ret = collect_all('pywebview')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

# Add hidden imports for common issues
hiddenimports.extend([
    'streamlit',
    'streamlit.web.bootstrap',
    'streamlit.web.cli',
    'streamlit.runtime',
    'streamlit.runtime.app_session',
    'streamlit.components',
    'cv2',
    'numpy',
    'numpy.core',
    'numpy.core.multiarray',
    'numpy.core.umath',
    'websocket',
    'websocket_client',
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    'PIL',
    'PIL._tkinter_finder',
    'PIL._imaging',
    'PIL._tkinter',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tensorflow',
    'tensorflow._api',
    'tensorflow._api.v2',
    'tensorflow.python',
    'tensorflow.python.framework',
    'tensorflow.python.keras',
    'tensorflow.python.keras.utils',
    'pywebview',
    'pywebview.platforms',
    'pywebview.platforms.winforms',
    'pywebview.platforms.edgechromium',
    'pywebview.platforms.mshtml',
    'pywebview.platforms.cef',
    'pywebview.platforms.qt',
    'pywebview.platforms.gtk',
    'pywebview.platforms.cocoa',
    'pkg_resources.py2_warn',
    'pkg_resources',
    'altair',
    'pandas',
])

# Add streamlit data files
try:
    datas += collect_data_files('streamlit')
except:
    pass

a = Analysis(
    ['desktop_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'ipykernel',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Build with console enabled first to see errors
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TFLiteTraining',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX to avoid compression issues
    console=True,  # Enable console to see errors
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# For debugging, let's make a single exe or directory
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.datas,
#     strip=False,
#     upx=False,
#     upx_exclude=[],
#     name='TFLiteTraining',
# )

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='TFLiteTraining.app',
        icon=None,
        bundle_identifier='ai.tflite.training',
        info_plist={
            'NSCameraUsageDescription': 'TFLiteTraining needs camera access to capture training samples.',
            'NSHighResolutionCapable': True,
        },
    )
