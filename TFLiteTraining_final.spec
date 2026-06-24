# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Data files to include
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

# Collect streamlit data (static files)
try:
    datas += collect_data_files('streamlit', include_py_files=True)
except Exception as e:
    print(f"Warning: Could not collect streamlit data: {e}")

try:
    datas += collect_data_files('streamlit_drawable_canvas', include_py_files=True)
except Exception as e:
    print(f"Warning: Could not collect streamlit_drawable_canvas data: {e}")

# Hidden imports
hiddenimports = [
    'streamlit.web.bootstrap',
    'streamlit.web.cli',
    'streamlit.runtime',
    'streamlit.runtime.app_session',
    'streamlit.runtime.scriptrunner',
    'streamlit.components',
    'streamlit.components.v1',
    'streamlit.runtime.forward_msg_cache',
    'streamlit.proto.BackMsg_pb2',
    'streamlit.proto.ClientState_pb2',
    'streamlit.proto.Common_pb2',
    'streamlit.proto.Components_pb2',
    'streamlit.proto.Element_pb2',
    'streamlit.proto.ForwardMsg_pb2',
    'streamlit.proto.Declare_component_pb2',
    'streamlit.proto.WidgetStates_pb2',
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
    'tkinter',
    'tkinter.filedialog',
    'altair',
    'pandas',
]

# Add streamlit submodules
try:
    hiddenimports += collect_submodules('streamlit')
except Exception as e:
    print(f"Warning: Could not collect streamlit submodules: {e}")

# Analysis
a = Analysis(
    ['desktop_launcher_simple.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'ipykernel', 'notebook'],
    noarchive=False,
    optimize=0,
)

# Build
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
