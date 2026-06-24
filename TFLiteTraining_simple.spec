# -*- mode: python ; coding: utf-8 -*-
import sys
import os

# A simpler, more reliable spec file

datas = [
    ('app.py', '.'),
    ('camera_permission.py', '.'),
    ('dataset_io.py', '.'),
    ('trainer.py', '.'),
    ('ui_styles.py', '.'),
    ('serial_device.py', '.'),
    ('record_controller.py', '.'),
    ('device_stream.py', '.'),
    ('tflite_train.py', '.'),
]

a = Analysis(
    ['desktop_launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'streamlit',
        'streamlit.web.bootstrap',
        'streamlit.runtime',
        'cv2',
        'numpy',
        'numpy.core',
        'numpy.core.multiarray',
        'websocket',
        'websocket_client',
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
        'PIL',
        'PIL._tkinter_finder',
        'PIL._imaging',
        'tensorflow',
        'pywebview',
        'pywebview.platforms',
        'pywebview.platforms.winforms',
        'pywebview.platforms.edgechromium',
        'tkinter',
        'tkinter.filedialog',
        'altair',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

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
    upx=False,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
