# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all

datas = [('app.py', '.'), ('camera_permission.py', '.'), ('dataset_io.py', '.'), ('trainer.py', '.'), ('ui_styles.py', '.'), ('serial_device.py', '.'), ('record_controller.py', '.'), ('image_preprocess.py', '.')]
binaries = []
hiddenimports = []

# macOS multiprocessing support — critical for frozen apps using spawn
hiddenimports += [
    'multiprocessing',
    'multiprocessing.process',
    'multiprocessing.queues',
    'multiprocessing.synchronize',
    'multiprocessing.pool',
    'multiprocessing.spawn',
    '_multiprocessing',
    'ctypes',
    'ctypes.macholib',
]

tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('streamlit_drawable_canvas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('serial')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

if sys.platform == 'darwin':
    tmp_ret = collect_all('pywebview')
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['desktop_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TFLiteTraining',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                     # UPX can corrupt macOS dylibs
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,           # needed for macOS .app file-open events
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
exe_console = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TFLiteTrainingConsole',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    exe_console,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TFLiteTraining',
)
app = BUNDLE(
    coll,
    name='TFLiteTraining.app',
    icon=None,
    bundle_identifier='ai.tflite.training',
    info_plist={
        'NSCameraUsageDescription': 'TFLiteTraining needs camera access to capture training samples.',
        'NSHighResolutionCapable': True,
    },
)
