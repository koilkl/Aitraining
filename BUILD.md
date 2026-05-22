# 使用与打包（macOS .dmg / Windows .exe）

目标：学生不需要安装 Python，双击即可打开。

## 软件怎么用（给学生/老师）

- 打开软件后依次使用：
  - New Project → Open Image Project
  - Import：导入 ZIP / 多张图片 / 本地文件夹
  - Label：为图片选择/确认类别，并保存为训练数据集
  - Train：设置超参数并训练（会生成 int8 TFLite）
  - Export：选择导出目录，导出 `model.h / model.cpp / .tflite`
- 数据输出目录（默认）：
  - macOS：`~/Library/Application Support/TFLiteTraining/`
  - Windows：`%APPDATA%\\TFLiteTraining\\`
  - 目录结构：
    - `datasets/`：你在 Label 步骤保存的训练数据集
    - `runs/`：每次 Train 产生的训练记录与模型产物
    - `logs/streamlit.log`：软件启动/运行日志（排错用）
- 导出到 MCU：
  - `model.h / model.cpp`：直接放进 Arduino/ESP-IDF 工程里（数组名可在 Export 步骤设置）
  - `labels.txt`：类别顺序（推理结果 index 对应）

## 目录说明（开发者）

- 桌面入口：`desktop_launcher.py`（启动本地 Streamlit 服务 + WebView 窗口）
- Streamlit 页面：`app.py`
- 数据导入/导出：`dataset_io.py`
- 训练/量化/导出：`trainer.py`

## macOS（生成 .app + .dmg）

```bash
cd AItraining
python3.11 -m venv .venv
source .venv/bin/activate
python3 -V

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt

python3 -m PyInstaller --noconfirm --windowed --name TFLiteTraining \
  --add-data "app.py:." \
  --add-data "dataset_io.py:." \
  --add-data "trainer.py:." \
  --add-data "ui_styles.py:." \
  --add-data "serial_device.py:." \
  --add-data "record_controller.py:." \
  --collect-all streamlit \
  --collect-all streamlit_drawable_canvas \
  --collect-all cv2 \
  desktop_launcher.py

dmgbuild -s dmg_settings.py "TFLiteTraining" "dist/TFLiteTraining.dmg"
```

产物：
- `dist/TFLiteTraining.app`
- `dist/TFLiteTraining.dmg`

## Windows（生成 .exe）

在 Windows 上执行：

```powershell
cd AItraining
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version

py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt

python -m PyInstaller --noconfirm --windowed --name TFLiteTraining `
  --add-data "app.py;." `
  --add-data "dataset_io.py;." `
  --add-data "trainer.py;." `
  --add-data "ui_styles.py;." `
  --add-data "serial_device.py;." `
  --add-data "record_controller.py;." `
  --collect-all streamlit `
  --collect-all streamlit_drawable_canvas `
  --collect-all cv2 `
  desktop_launcher.py
```

产物：
- `dist\\TFLiteTraining\\TFLiteTraining.exe`

## 常见问题

- macOS “无法打开/来自不明开发者”：右键 `TFLiteTraining.app` → 打开 → 确认；或在系统设置里允许。
- Windows 运行缺 DLL：安装 Microsoft Visual C++ Redistributable 2015–2022 (x64)。
