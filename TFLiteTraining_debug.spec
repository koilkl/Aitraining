# -*- mode: python ; coding: utf-8 -*-
"""
Debug spec file - keeps console open and pauses on errors
"""
import sys
import os

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
    ('desktop_launcher.py', '.'),
]

a = Analysis(
    ['desktop_launcher_debug.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'streamlit',
        'streamlit.web.bootstrap',
        'streamlit.runtime',
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
        'tensorflow',
        'tensorflow._api',
        'tensorflow._api.v2',
        'pywebview',
        'pywebview.platforms',
        'pywebview.platforms.winforms',
        'pywebview.platforms.edgechromium',
        'pywebview.platforms.mshtml',
        'pywebview.platforms.cef',
        'pywebview.platforms.qt',
        'pywebview.platforms.gtk',
        'pywebview.platforms.cocoa',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'pkg_resources',
        'altair',
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        'pandas._libs.tslibs.np_datetime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'ipykernel', 'notebook'],
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
    name='TFLiteTraining_Debug',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
