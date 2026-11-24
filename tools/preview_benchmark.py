"""Measure the 04 PREVIEW backend latency (before/after the decode cache).

The preview tick used to decode the camera PNG on EVERY request.  The live
route now decodes once per produced frame and reuses that image, so the
two columns below are:

    cold  raw PNG bytes into the predictor  (the old per-tick decode)
    warm  the already-decoded PIL frame     (what the route now passes)

`total` is the predictor call; the HTTP route adds ~1 ms of localhost
overhead and the unchanged-frame short circuit (~0.02 ms) when the
producer has not published a new frame.

Run:
    .venv/bin/python tools/preview_benchmark.py
"""
from __future__ import annotations

import io
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import record_controller as rc  # noqa: E402

TEMPLATE_PROJECT = ROOT / "templates" / "car_sign_template.tmproj"
SEQUENCE = ["A", "1", "2"]


def _workspace(root: Path):
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    ctrl = rc.RecordController()
    ctrl._configs["s"] = rc.SessionConfig(
        dataset_root=dataset,
        serial_port="",
        serial_baud=115200,
        serial_sync="AA 55 AA",
        serial_frame_side=96,
        serial_channels=1,
        webcam_index=0,
        fps=8.0,
        crop_box=None,
        img_size=28,
        project_type=rc.PROJECT_TYPE_SIGN_OCR,
    )
    ctrl._classes_save(dataset, list(SEQUENCE))
    for name in SEQUENCE:
        (dataset / name).mkdir(exist_ok=True)
    state = ctrl._project_open("s", TEMPLATE_PROJECT)
    ctrl._train_worker("s", state["train_cfg"])
    assert not (ctrl._train.get("s") or {}).get("error"), ctrl._train.get("s")
    return ctrl


def _scene_with_plate(plate_png: bytes, size=(640, 480)) -> Image.Image:
    plate = Image.open(io.BytesIO(plate_png)).convert("L")
    plate = plate.resize((240, 96))
    width, height = size
    rng = np.random.default_rng(1)
    frame = np.clip(rng.normal(110, 10, (height, width)), 0, 255).astype(np.uint8)
    y0 = (height - 96) // 2
    x0 = (width - 240) // 2
    frame[y0:y0 + 96, x0:x0 + 240] = np.asarray(plate)
    return Image.fromarray(np.stack([frame] * 3, axis=-1))


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stats(times):
    ordered = sorted(times)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def _bench(fn, n=25):
    for _ in range(3):
        fn()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return _stats(times)


def _image_controller(dataset_root: Path, side: int):
    class _Stub:
        def set_tensor(self, *_a):
            pass

        def invoke(self):
            pass

        def get_tensor(self, _i):
            return np.array([[0, 127]], dtype=np.int8)

    ctrl = rc.RecordController()
    session = f"img{side}"
    ctrl._configs[session] = SimpleNamespace(
        dataset_root=dataset_root,
        direct_tensor_stream=False,
        project_type="image",
    )
    ctrl._preview[session] = {
        "interpreter": _Stub(),
        "input_details": {
            "shape": np.array([1, side, side, 1]),
            "dtype": np.int8,
            "quantization": (1 / 255, -128),
            "index": 0,
        },
        "output_details": {"dtype": np.int8, "quantization": (1 / 255, -128), "index": 1},
        "labels": ["A", "B"],
        "preprocess_mode": rc.PREPROCESS_MODE_AUTO_BY_LABEL,
        "crop_mode": rc.CROP_MODE_AUTO_SEARCH,
        "manual_roi": None,
        "class_preprocess": {},
        "model_source": "latest",
    }
    ctrl._preview_load_model = lambda session_id=None, **kw: ctrl._preview[session]
    return ctrl, session


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ctrl = _workspace(Path(td))
        synth = ctrl._synthesize_sign_preview_png(
            "s", variant="match", sign_sequence="A12", reading_order="ltr"
        )
        scene = _scene_with_plate(synth["png"])
        scene_bytes = _png_bytes(scene)
        print(f"frame: {scene.size[0]}x{scene.size[1]} · compact JPEG q82")
        print()
        print(f"{'path':34s} {'mode':6s} {'median':>8s} {'p95':>8s}")
        print("-" * 60)

        for label, kwargs in (
            ("sign_ocr ROI off", dict(view_roi=False, view_orig=False)),
            ("sign_ocr ROI on (3-slot strip)", dict(view_roi=True, view_orig=False)),
        ):
            cold = _bench(lambda: ctrl._preview_predict_sign_ocr(
                "s", scene_bytes, sign_sequence="A12", sign_sequences=["A12"],
                reading_order="ltr", source="webcam", compact=True, **kwargs))
            warm = _bench(lambda: ctrl._preview_predict_sign_ocr(
                "s", scene, sign_sequence="A12", sign_sequences=["A12"],
                reading_order="ltr", source="webcam", compact=True, **kwargs))
            for mode, (median, p95) in (("cold", cold), ("warm", warm)):
                print(f"{label:34s} {mode:6s} {median:7.2f}ms {p95:7.2f}ms")

        rgb_scene = scene
        # The image classifier is a 96×96 INT8 model (the app default), so
        # the live webcam path searches a frame downscaled to that side.
        for frame in (rgb_scene, rgb_scene.resize((96, 96))):
            frame_bytes = _png_bytes(frame)
            img_ctrl, session = _image_controller(Path(td), 96)
            for kwargs in (dict(view_roi=False), dict(view_roi=True)):
                cold = _bench(lambda: img_ctrl._preview_predict(
                    session_id=session, png=frame_bytes, source="webcam", compact=True, **kwargs))
                warm = _bench(lambda: img_ctrl._preview_predict(
                    session_id=session, png=frame, source="webcam", compact=True, **kwargs))
                label = f"image {frame.size[0]}x{frame.size[1]} {'ROI off' if not kwargs['view_roi'] else 'ROI on'}"
                for mode, (median, p95) in (("cold", cold), ("warm", warm)):
                    print(f"{label:34s} {mode:6s} {median:7.2f}ms {p95:7.2f}ms")
        print()
        print("Note: live webcam frames are searched at the model side, the")
        print("device's own order (camera -> downsize -> crop search).")
        print("Uploads keep full-resolution search (their training cache is")
        print("full-res), which is the bit-exact but slower ~50 ms path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
