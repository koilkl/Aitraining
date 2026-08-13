# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all

datas = [('app.py', '.'), ('camera_permission.py', '.'), ('dataset_io.py', '.'), ('trainer.py', '.'), ('ui_styles.py', '.'), ('serial_device.py', '.'), ('record_controller.py', '.'), ('image_preprocess.py', '.'), ('file_dialog.py', '.')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('streamlit_drawable_canvas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tensorflow')
# Exclude the C++ headers under tensorflow/include — they are dev-only, bloat
# the bundle by ~1 GB, and their deep paths exceed Windows MAX_PATH in the
# installer (Inno Setup fails with "系统找不到指定的路径").
tf_datas = [d for d in tmp_ret[0] if '/tensorflow/include/' not in str(d[0]).replace('\\', '/')]
datas += tf_datas; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('serial')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Windows: ensure pyserial backend is collected
hiddenimports += ['serial.tools.list_ports_windows', 'serial.tools.list_ports_common']

# Windows: multiprocessing support for frozen apps
hiddenimports += ['multiprocessing', 'multiprocessing.process', 'multiprocessing.spawn',
                   'multiprocessing.queues', 'multiprocessing.synchronize',
                   'multiprocessing.pool', 'multiprocessing.heap',
                   'multiprocessing.reduction', 'multiprocessing.resource_tracker']

# Ensure pathlib is available (resolved Path objects)
hiddenimports += ['pathlib']

# Dynamically imported modules used on Windows/macOS
hiddenimports += ['webview', 'pywebview', 'tkinter', 'websocket', 'websocket-client',
                   'PIL', 'PIL.Image', 'io', 'uuid', 'json', 'base64']


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
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TFLiteTraining',
)

# macOS: bundle into .app; Windows: COLLECT alone produces a dist folder with the .exe
if sys.platform == "darwin":
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
