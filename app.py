from __future__ import annotations

import shutil
import uuid
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from trainer import TrainConfig, new_run_dir, train_and_export
from ui_styles import inject_teachable_style

from dataset_io import (
    IMAGE_EXTS,
    ImportedData,
    export_classified_dataset,
    export_from_assignments,
    infer_imported_data,
    sanitize_class_name,
    materialize_files,
    materialize_zip_bytes,
    new_output_dir,
)
from serial_device import list_serial_ports, read_frame_png_from_serial
from record_controller import RecordController, SessionConfig, make_hold_button_html


APP_NAME = "TFLiteTraining"


def _app_data_dir() -> Path:
    env_override = os.getenv("TFLITE_TRAINING_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / APP_NAME).resolve()


APP_DATA_DIR = _app_data_dir()
WORKSPACE_DIR = APP_DATA_DIR / "workspace"
DATASETS_DIR = APP_DATA_DIR / "datasets"


def _documents_dir() -> Path:
    return (Path.home() / "Documents").resolve()


def _default_export_dir() -> Path:
    return (_documents_dir() / APP_NAME / "exports").resolve()


def _pick_directory_dialog(initial_dir: Optional[str] = None) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askdirectory(initialdir=initial_dir or str(_documents_dir()))
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return path or None


def _validate_export_inputs(export_dir: Path, model_name: str, array_name: str, tflite_path: Path) -> List[str]:
    errors: List[str] = []
    if not model_name.strip():
        errors.append("模型文件名前缀不能为空")
    if any(sep in model_name for sep in ("/", "\\", os.sep)):
        errors.append("模型文件名前缀不能包含路径分隔符")
    if not re.match(r"^[A-Za-z0-9_.-]+$", model_name):
        errors.append("模型文件名前缀仅允许字母/数字/._-")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", array_name):
        errors.append("C 数组名必须是合法标识符（字母/数字/下划线，且不能以数字开头）")
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"无法创建导出目录：{export_dir} ({e})")
        return errors
    if not export_dir.is_dir():
        errors.append(f"导出目录不是文件夹：{export_dir}")
    if not tflite_path.exists():
        errors.append("找不到 .tflite 文件，请先完成训练")
    try:
        test_file = export_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        errors.append(f"导出目录不可写：{export_dir} ({e})")
    return errors


def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "imported" not in st.session_state:
        st.session_state.imported = None
    if "project_type" not in st.session_state:
        st.session_state.project_type = None
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "dataset_dir" not in st.session_state:
        st.session_state.dataset_dir = None
    if "assignments" not in st.session_state:
        st.session_state.assignments = {}
    if "class_names" not in st.session_state:
        st.session_state.class_names = ["Class 1", "Class 2"]
    if "class_rename" not in st.session_state:
        st.session_state.class_rename = {}
    if "last_export_dir" not in st.session_state:
        st.session_state.last_export_dir = str(_default_export_dir())
    if "train_cfg" not in st.session_state:
        st.session_state.train_cfg = TrainConfig()
    if "train_result" not in st.session_state:
        st.session_state.train_result = None
    if "local_import_path" not in st.session_state:
        st.session_state.local_import_path = ""
    if "export_validated_token" not in st.session_state:
        st.session_state.export_validated_token = ""
    if "tm_classes" not in st.session_state:
        st.session_state.tm_classes = ["Class 1", "Class 2"]
    if "tm_device_url" not in st.session_state:
        st.session_state.tm_device_url = ""
    if "tm_device_enabled" not in st.session_state:
        st.session_state.tm_device_enabled = False
    if "tm_serial_port" not in st.session_state:
        st.session_state.tm_serial_port = ""
    if "tm_serial_baud" not in st.session_state:
        st.session_state.tm_serial_baud = 115200
    if "tm_last_device_frame" not in st.session_state:
        st.session_state.tm_last_device_frame = None
    if "tm_capture_open" not in st.session_state:
        st.session_state.tm_capture_open = False
    if "tm_capture_source" not in st.session_state:
        st.session_state.tm_capture_source = ""
    if "tm_capture_class" not in st.session_state:
        st.session_state.tm_capture_class = ""
    if "tm_record_fps" not in st.session_state:
        st.session_state.tm_record_fps = 8.0
    if "tm_webcam_index" not in st.session_state:
        st.session_state.tm_webcam_index = 0
    if "tm_record_crop_box" not in st.session_state:
        st.session_state.tm_record_crop_box = None
    if "tm_crop_mode" not in st.session_state:
        st.session_state.tm_crop_mode = "full"
    if "tm_pending_image" not in st.session_state:
        st.session_state.tm_pending_image = None
    if "tm_pending_class" not in st.session_state:
        st.session_state.tm_pending_class = None


def _session_workspace() -> Path:
    p = WORKSPACE_DIR / st.session_state.session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reset_session_workspace() -> None:
    p = WORKSPACE_DIR / st.session_state.session_id
    if p.exists():
        shutil.rmtree(p)
    st.session_state.imported = None
    st.session_state.project_type = None
    st.session_state.step = 0
    st.session_state.dataset_dir = None
    st.session_state.assignments = {}
    st.session_state.class_names = ["Class 1", "Class 2"]
    st.session_state.class_rename = {}
    st.session_state.train_cfg = TrainConfig()
    st.session_state.train_result = None
    st.session_state.export_validated_token = ""
    st.session_state.tm_classes = ["Class 1", "Class 2"]
    st.session_state.tm_device_enabled = False
    st.session_state.tm_crop_mode = "full"
    st.session_state.tm_pending_image = None
    st.session_state.tm_pending_class = None
    st.session_state.tm_last_device_frame = None
    st.session_state.tm_capture_open = False
    st.session_state.tm_capture_source = ""
    st.session_state.tm_capture_class = ""
    st.session_state.tm_record_crop_box = None


def _read_uploaded_images(uploaded_files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        out.append((f.name, f.getvalue()))
    return out


def _render_import_panel() -> None:
    st.subheader("导入图片")
    method = st.radio("导入方式", ["ZIP 压缩包", "多张图片", "本地文件夹路径"], horizontal=True)
    ws = _session_workspace()

    if method == "ZIP 压缩包":
        zip_file = st.file_uploader("拖入或选择 .zip", type=["zip"])
        if zip_file is not None and st.button("导入", type="primary"):
            dest = ws / "import"
            materialize_zip_bytes(zip_file.getvalue(), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    elif method == "多张图片":
        exts = sorted([e.lstrip(".") for e in IMAGE_EXTS])
        files = st.file_uploader("从文件夹中全选图片后拖入（或多选）", type=exts, accept_multiple_files=True)
        if files and st.button("导入", type="primary"):
            dest = ws / "import"
            materialize_files(_read_uploaded_images(files), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    else:
        left, right = st.columns([4, 1])
        with left:
            path_str = st.text_input(
                "输入本地文件夹路径（可已分类或未分类）",
                value=st.session_state.local_import_path,
            )
            st.session_state.local_import_path = path_str
        with right:
            if st.button("浏览...", key="browse_import_dir"):
                picked = _pick_directory_dialog(initial_dir=st.session_state.local_import_path)
                if picked:
                    st.session_state.local_import_path = picked
                    st.rerun()

        if st.button("读取路径", type="primary"):
            path_str = st.session_state.local_import_path
            p = Path(path_str).expanduser()
            if not p.exists() or not p.is_dir():
                st.error("路径不存在或不是文件夹")
                return
            st.session_state.imported = infer_imported_data(p)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}


def _render_overview(imported: ImportedData) -> None:
    st.subheader("预览")
    st.markdown(
        f"""
<div class="tm-kv">
  <b>导入目录</b>：{imported.root_dir}<br/>
  <b>图片数量</b>：{len(imported.images)}<br/>
  <b>已按文件夹分类</b>：{imported.classified}
</div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    sample = imported.images[: min(len(imported.images), 6)]
    for c, p in zip(cols, sample):
        with c:
            st.image(str(p), use_container_width=True)
            st.caption(p.name)


def _render_classified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("Class 重命名（已分类数据）")
    class_names = list(imported.class_to_images.keys())
    st.write({"检测到的 class": class_names})

    rename: Dict[str, str] = {}
    for c in class_names:
        rename[c] = st.text_input(f"{c} →", value=st.session_state.class_rename.get(c, c), key=f"rename_{c}")
    st.session_state.class_rename = rename

    st.write({c: len(imported.class_to_images[c]) for c in class_names})

    if st.button("保存为训练数据集", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="classified")
        try:
            export_classified_dataset(imported.class_to_images, rename, out_dir)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"已生成训练数据集：{out_dir}")
        return out_dir
    return None


def _render_unclassified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("选择/Clarify Class（未分类数据）")
    st.session_state.class_names = _render_class_editor(st.session_state.class_names)
    class_names = st.session_state.class_names

    if not class_names:
        st.warning("请先创建至少 1 个 class")
        return None

    page_size = st.slider("每页显示图片数", min_value=6, max_value=60, value=18, step=6)
    total = len(imported.images)
    max_page = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("页码", min_value=1, max_value=max_page, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)

    for idx in range(start, end):
        img = imported.images[idx]
        left, right = st.columns([1, 2])
        with left:
            st.image(str(img), use_container_width=True)
        with right:
            default = st.session_state.assignments.get(idx)
            options = ["未分配"] + class_names
            try:
                default_index = options.index(default) if default in options else 0
            except Exception:
                default_index = 0
            choice = st.selectbox(
                f"[{idx}] {img.name}",
                options=options,
                index=default_index,
                key=f"assign_{idx}",
            )
            if choice == "未分配":
                st.session_state.assignments.pop(idx, None)
            else:
                st.session_state.assignments[idx] = choice

    require_all = st.checkbox("保存时要求全部图片已分配类别", value=True)
    assigned_count = len(st.session_state.assignments)
    st.write({"已分配": assigned_count, "未分配": total - assigned_count})

    if st.button("保存为训练数据集", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="labeled")
        assignments: List[Optional[str]] = []
        for i in range(total):
            assignments.append(st.session_state.assignments.get(i))
        try:
            export_from_assignments(imported.images, assignments, out_dir, require_all_assigned=require_all)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"已生成训练数据集：{out_dir}")
        return out_dir

    return None


def _render_class_editor(class_names: List[str]) -> List[str]:
    st.write("Class 列表（每行一个）")
    txt = st.text_area(" ", value="\n".join(class_names), height=140, key="class_editor")
    names = [x.strip() for x in txt.splitlines()]
    names = [x for x in names if x]
    return names


def _render_steps() -> None:
    steps = ["Import", "Label", "Train", "Export"]
    s = int(st.session_state.step)
    parts: List[str] = []
    for i, name in enumerate(steps):
        if i == s:
            parts.append(f'<span class="tm-step-on">{i+1}. {name}</span>')
        else:
            parts.append(f"{i+1}. {name}")
    st.markdown(f'<div class="tm-steps">{"  ·  ".join(parts)}</div>', unsafe_allow_html=True)


def _render_new_project() -> None:
    inject_teachable_style()
    st.markdown('<div class="tm-title">New Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="tm-sub">Create a project to teach from your examples.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="tm-card">
  <h3>Image Project</h3>
  <p>Teach based on images, from files or your camera.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Image Project", type="primary"):
            st.session_state.project_type = "image"
            st.session_state.step = 0
            st.rerun()
    with c2:
        st.markdown(
            """
<div class="tm-card">
  <h3>Audio Project <span class="tm-badge">Coming soon</span></h3>
  <p>Teach based on sound samples from microphone.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="tm-card">
  <h3>Pose Project <span class="tm-badge">Coming soon</span></h3>
  <p>Teach based on poses from images.</p>
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_train_config(cfg: TrainConfig) -> TrainConfig:
    left, right = st.columns(2)
    with left:
        img_size = st.number_input("Image size", min_value=32, max_value=224, value=int(cfg.img_size), step=8)
        color_mode = st.selectbox("Color mode", options=["rgb", "grayscale"], index=0 if cfg.color_mode == "rgb" else 1)
        batch_size = st.selectbox("Batch size", options=[8, 16, 32, 64], index=[8, 16, 32, 64].index(cfg.batch_size))
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(cfg.epochs), step=1)
        validation_split = st.slider("Validation split", min_value=0.05, max_value=0.5, value=float(cfg.validation_split), step=0.05)
    with right:
        optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=["adam", "sgd", "rmsprop"].index(cfg.optimizer.lower()))
        learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=float(cfg.learning_rate), format="%.5f")
        conv1_filters = st.selectbox("Conv1 filters", options=[4, 8, 16, 32], index=[4, 8, 16, 32].index(cfg.conv1_filters))
        conv2_filters = st.selectbox("Conv2 filters", options=[8, 16, 32, 64], index=[8, 16, 32, 64].index(cfg.conv2_filters))
        dense_units = st.selectbox("Dense units", options=[16, 32, 64, 128], index=[16, 32, 64, 128].index(cfg.dense_units))

    return TrainConfig(
        img_size=int(img_size),
        color_mode=str(color_mode),
        batch_size=int(batch_size),
        epochs=int(epochs),
        validation_split=float(validation_split),
        seed=int(cfg.seed),
        optimizer=str(optimizer),
        learning_rate=float(learning_rate),
        conv1_filters=int(conv1_filters),
        conv2_filters=int(conv2_filters),
        dense_units=int(dense_units),
        representative_samples=int(cfg.representative_samples),
    )


def _render_image_project() -> None:
    inject_teachable_style()
    st.markdown('<div class="tm-title">Image Project</div>', unsafe_allow_html=True)
    nav_left, nav_right = st.columns([1, 4])
    with nav_left:
        if st.button("← Back"):
            _reset_session_workspace()
            st.rerun()

    left, mid, right = st.columns([2.3, 1.2, 2.1], gap="large")
    with left:
        _render_tm_class_panel()
    with mid:
        _render_tm_train_panel()
    with right:
        _render_tm_preview_export_panel()


def _tm_dataset_dir() -> Path:
    ws = _session_workspace()
    return (ws / "tm_dataset").resolve()


@st.cache_resource
def _get_record_controller() -> RecordController:
    c = RecordController()
    c.start()
    return c


def _save_sample_png(class_name: str, png_bytes: bytes) -> Path:
    out_dir = _tm_dataset_dir() / sanitize_class_name(class_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".png"
    out_path = out_dir / name
    out_path.write_bytes(png_bytes)
    return out_path


def _img_to_png_bytes(img) -> bytes:
    import io
    from PIL import Image

    if isinstance(img, Image.Image):
        im = img
    else:
        im = Image.open(io.BytesIO(img))
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _preprocess_image_to_96x96_gray(png_bytes: bytes, crop_box: Optional[Tuple[int, int, int, int]] = None) -> bytes:
    import io
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        im = im.crop((x1, y1, x2, y2))
    im = im.resize((96, 96))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _render_crop_ui() -> None:
    if st.session_state.tm_pending_image is None:
        return
    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception:
        st.error("缺少依赖 streamlit-drawable-canvas，无法使用框选裁剪模式")
        return

    import io
    from PIL import Image

    st.subheader("Crop 区域（框选）")
    png_bytes = st.session_state.tm_pending_image
    class_name = st.session_state.tm_pending_class
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    im_disp = im.resize((disp_w, disp_h))

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="rgba(0, 122, 255, 1)",
        background_image=im_disp,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key="crop_canvas",
    )

    rect = None
    if canvas.json_data and canvas.json_data.get("objects"):
        obj = canvas.json_data["objects"][-1]
        if obj and obj.get("type") == "rect":
            rect = obj

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("取消", key="crop_cancel"):
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()
    with col_b:
        if st.button("确认保存", type="primary", key="crop_confirm"):
            if rect is None:
                st.error("请先画一个矩形框")
                return
            left = int(rect.get("left", 0))
            top = int(rect.get("top", 0))
            rw = int(rect.get("width", 0))
            rh = int(rect.get("height", 0))
            if rw <= 1 or rh <= 1:
                st.error("框选区域太小")
                return
            x1 = int(left / scale)
            y1 = int(top / scale)
            x2 = int((left + rw) / scale)
            y2 = int((top + rh) / scale)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            out_png = _preprocess_image_to_96x96_gray(png_bytes, crop_box=(x1, y1, x2, y2))
            _save_sample_png(class_name, out_png)
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()

    st.caption(f"原图：{w}x{h}，输出：96x96x1")


def _render_tm_class_panel() -> None:
    st.markdown("### Classes")

    mode = st.radio("输入模式", ["原图缩放到 96x96x1", "框选 ROI 后缩放到 96x96x1"], horizontal=False)
    st.session_state.tm_crop_mode = "roi" if "ROI" in mode else "full"

    with st.expander("Bulk import（可选）", expanded=False):
        _render_import_panel()
        imported = st.session_state.imported
        if imported is not None:
            _render_overview(imported)
            out: Optional[Path]
            if imported.classified:
                out = _render_classified_flow(imported)
            else:
                out = _render_unclassified_flow(imported)
            if out is not None:
                st.session_state.dataset_dir = str(out)

    _render_crop_ui()

    classes_txt = st.text_area("Class 列表（每行一个）", value="\n".join(st.session_state.tm_classes), height=120, key="tm_class_list")
    classes = [x.strip() for x in classes_txt.splitlines() if x.strip()]
    if not classes:
        classes = ["Class 1"]
    st.session_state.tm_classes = classes

    controller = _get_record_controller()
    controller.set_config(
        st.session_state.session_id,
        SessionConfig(
            dataset_root=_tm_dataset_dir(),
            serial_port=st.session_state.tm_serial_port,
            serial_baud=int(st.session_state.tm_serial_baud),
            webcam_index=int(st.session_state.tm_webcam_index),
            fps=float(st.session_state.tm_record_fps),
            crop_box=st.session_state.tm_record_crop_box,
        ),
    )

    for idx, name in enumerate(classes):
        st.markdown(f"#### {name}")
        class_dir = _tm_dataset_dir() / sanitize_class_name(name)
        samples = sorted([p for p in class_dir.glob("*.png")]) if class_dir.exists() else []
        st.write({"samples": len(samples)})

        upload = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True, key=f"tm_up_{idx}")
        if upload:
            for f in upload:
                png = _img_to_png_bytes(f.getvalue())
                if st.session_state.tm_crop_mode == "roi":
                    st.session_state.tm_pending_image = png
                    st.session_state.tm_pending_class = name
                    st.rerun()
                out_png = _preprocess_image_to_96x96_gray(png, crop_box=None)
                _save_sample_png(name, out_png)
            st.rerun()

        btn_a, btn_b, btn_c, btn_d = st.columns([1, 1, 1, 3])
        with btn_a:
            if st.button("Webcam", key=f"tm_open_webcam_{idx}"):
                st.session_state.tm_capture_open = True
                st.session_state.tm_capture_source = "webcam"
                st.session_state.tm_capture_class = name
                st.rerun()
        with btn_b:
            if st.button("Device", key=f"tm_open_device_{idx}"):
                st.session_state.tm_capture_open = True
                st.session_state.tm_capture_source = "device"
                st.session_state.tm_capture_class = name
                st.rerun()
        with btn_c:
            if st.button("Close", key=f"tm_close_capture_{idx}"):
                if st.session_state.tm_capture_class == name:
                    st.session_state.tm_capture_open = False
                    st.session_state.tm_capture_source = ""
                    st.session_state.tm_capture_class = ""
                    st.rerun()
        with btn_d:
            if samples:
                st.image([str(p) for p in samples[-6:]], width=64)

        if st.session_state.tm_capture_open and st.session_state.tm_capture_class == name:
            _render_hold_capture_panel(controller, name)


def _render_hold_capture_panel(controller: RecordController, class_name: str) -> None:
    from urllib.parse import quote

    source = st.session_state.tm_capture_source
    st.markdown('<div class="tm-panel">', unsafe_allow_html=True)
    st.markdown(f"**{source.upper()}**", unsafe_allow_html=True)

    preview: Optional[bytes] = None
    if source == "device":
        preview = controller.preview_serial_png(st.session_state.tm_serial_port, int(st.session_state.tm_serial_baud))
        if preview is None:
            st.warning("无法预览设备画面，请确认串口与波特率")
    elif source == "webcam":
        preview = controller.preview_webcam_png(int(st.session_state.tm_webcam_index))
        if preview is None:
            st.warning("无法预览摄像头画面，请检查系统权限/摄像头是否被占用")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    if st.session_state.tm_crop_mode == "roi" and preview is not None:
        try:
            from streamlit_drawable_canvas import st_canvas
        except Exception:
            st.error("缺少依赖 streamlit-drawable-canvas，无法使用框选裁剪模式")
        else:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(preview)).convert("RGB")
            w, h = im.size
            scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
            disp_w, disp_h = int(w * scale), int(h * scale)
            im_disp = im.resize((disp_w, disp_h))
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="rgba(0, 122, 255, 1)",
                background_image=im_disp,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="rect",
                key="tm_hold_roi_canvas",
            )
            if canvas.json_data and canvas.json_data.get("objects"):
                obj = canvas.json_data["objects"][-1]
                if obj and obj.get("type") == "rect":
                    left = int(obj.get("left", 0))
                    top = int(obj.get("top", 0))
                    rw = int(obj.get("width", 0))
                    rh = int(obj.get("height", 0))
                    if rw > 1 and rh > 1:
                        x1 = int(left / scale)
                        y1 = int(top / scale)
                        x2 = int((left + rw) / scale)
                        y2 = int((top + rh) / scale)
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(x1 + 1, min(w, x2))
                        y2 = max(y1 + 1, min(h, y2))
                        crop_box = (x1, y1, x2, y2)
                        st.session_state.tm_record_crop_box = crop_box

    if preview is not None:
        st.image(preview, use_container_width=True)

    st.session_state.tm_record_fps = st.slider("Record FPS", min_value=1.0, max_value=20.0, value=float(st.session_state.tm_record_fps), step=1.0)

    base = f"http://127.0.0.1:{controller.port}"
    q_class = quote(class_name)
    q_sess = quote(st.session_state.session_id)
    q_source = quote(source)
    start_url = f"{base}/start?session={q_sess}&source={q_source}&class={q_class}"
    stop_url = f"{base}/stop?session={q_sess}"
    html = make_hold_button_html("Hold to record", start_url=start_url, stop_url=stop_url)
    components.html(html, height=110)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_tm_train_panel() -> None:
    st.markdown("### Training")

    with st.expander("Advanced", expanded=False):
        st.session_state.train_cfg = _render_train_config(st.session_state.train_cfg)

    dataset_dir = _tm_dataset_dir()
    st.write({"dataset": str(dataset_dir)})
    if st.button("Train Model", type="primary"):
        st.session_state.export_validated_token = ""
        runs_dir = APP_DATA_DIR / "runs"
        run_dir = new_run_dir(runs_dir)
        cfg = st.session_state.train_cfg
        cfg = TrainConfig(
            img_size=96,
            color_mode="grayscale",
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_split=cfg.validation_split,
            seed=cfg.seed,
            optimizer=cfg.optimizer,
            learning_rate=cfg.learning_rate,
            conv1_filters=cfg.conv1_filters,
            conv2_filters=cfg.conv2_filters,
            dense_units=cfg.dense_units,
            representative_samples=cfg.representative_samples,
        )
        st.session_state.train_cfg = cfg
        with st.spinner("训练与导出 TFLite（int8）中..."):
            result = train_and_export(
                dataset_dir=dataset_dir,
                run_dir=run_dir,
                cfg=cfg,
                model_base_name="model",
                array_name="g_model",
            )
        st.session_state.train_result = result
        st.success(f"训练完成：val_acc={result.metrics.get('val_accuracy'):.4f}")


def _render_tm_preview_export_panel() -> None:
    st.markdown("### Preview")
    st.session_state.tm_webcam_index = st.selectbox("Webcam", options=[0, 1, 2], index=[0, 1, 2].index(int(st.session_state.tm_webcam_index)) if int(st.session_state.tm_webcam_index) in [0, 1, 2] else 0)
    ports = list_serial_ports()
    port_labels = [f"{p.device} ({p.description})" if p.description else p.device for p in ports]
    port_values = [p.device for p in ports]
    selected = st.selectbox(
        "Serial Port",
        options=[""] + port_values,
        format_func=lambda v: "请选择..." if v == "" else port_labels[port_values.index(v)],
        index=([""] + port_values).index(st.session_state.tm_serial_port) if st.session_state.tm_serial_port in ([""] + port_values) else 0,
    )
    st.session_state.tm_serial_port = selected
    st.session_state.tm_serial_baud = st.selectbox("Baudrate", options=[115200, 921600], index=0 if int(st.session_state.tm_serial_baud) == 115200 else 1)

    if st.button("Test Capture", key="tm_test_capture"):
        if not st.session_state.tm_serial_port:
            st.error("请先选择串口")
        else:
            try:
                png = read_frame_png_from_serial(
                    port=st.session_state.tm_serial_port,
                    baud=int(st.session_state.tm_serial_baud),
                    timeout_s=3.0,
                )
                st.session_state.tm_last_device_frame = png
                st.success("捕获成功")
            except Exception as e:
                st.error(str(e))

    if st.session_state.tm_last_device_frame:
        st.image(st.session_state.tm_last_device_frame, caption="Device frame (96x96)", use_container_width=True)
    else:
        st.info("选择串口并 Test Capture，或在左侧打开 Webcam/Device 面板录制。")

    st.markdown("### Export")
    result = st.session_state.train_result
    if result is None:
        st.info("Train a model before you can export.")
        return

    export_left, export_right = st.columns([4, 1])
    with export_left:
        export_dir_str = st.text_input(
            "导出到目录（默认：Documents/TFLiteTraining/exports）",
            value=st.session_state.last_export_dir,
        )
        st.session_state.last_export_dir = export_dir_str
    with export_right:
        if st.button("浏览...", key="tm_browse_export_dir"):
            picked = _pick_directory_dialog(initial_dir=st.session_state.last_export_dir)
            if picked:
                st.session_state.last_export_dir = picked
                st.rerun()

    model_name = st.text_input("模型文件名前缀", value="model", key="tm_model_name")
    array_name = st.text_input("C 数组名", value="g_model", key="tm_array_name")

    export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
    current_token = f"{export_dir}|{model_name}|{array_name}"
    validated = st.session_state.export_validated_token == current_token

    st.markdown("#### 验证")
    if st.button("Validate", key="tm_validate"):
        errors = _validate_export_inputs(export_dir, model_name, array_name, result.tflite_path)
        if errors:
            st.session_state.export_validated_token = ""
            st.error("\n".join(errors))
        else:
            st.session_state.export_validated_token = current_token
            st.success("验证通过，可以导出了")
            st.rerun()

    if st.button("Export Model", type="primary", key="tm_export", disabled=not validated):
        errors = _validate_export_inputs(export_dir, model_name, array_name, result.tflite_path)
        if errors:
            st.session_state.export_validated_token = ""
            st.error("\n".join(errors))
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        source_bytes = result.tflite_path.read_bytes()
        from trainer import export_tflite_c_sources

        src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
        (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
        (export_dir / "model.h").write_text(hdr, encoding="utf-8")
        (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
        (export_dir / "labels.txt").write_text("\n".join(result.labels) + "\n", encoding="utf-8")
        st.success(f"已导出到：{export_dir}")

    if st.button("Back to Train", key="tm_back_train"):
        st.session_state.train_result = None
        st.session_state.export_validated_token = ""
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="TF Lite Training", layout="wide")
    _init_session()

    if st.session_state.project_type is None:
        _render_new_project()
        return

    if st.session_state.project_type == "image":
        _render_image_project()
    else:
        st.warning("该项目类型暂未实现")


if __name__ == "__main__":
    main()
