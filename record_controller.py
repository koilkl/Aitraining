from __future__ import annotations

import base64
import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from serial_device import SerialFrameReader
from dataset_io import sanitize_class_name


@dataclass
class SessionConfig:
    dataset_root: Path
    serial_port: str
    serial_baud: int
    webcam_index: int
    fps: float
    crop_box: Optional[Tuple[int, int, int, int]]


class RecordController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configs: Dict[str, SessionConfig] = {}
        self._active: Dict[str, Dict[str, str]] = {}
        self._server: Optional[HTTPServer] = None
        self._port: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Controller server not started")
        return self._port

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            host = "127.0.0.1"
            port = _find_free_port(host)
            server = HTTPServer((host, port), self._make_handler())
            self._server = server
            self._port = int(port)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            self._thread = t
            t.start()

    def set_config(self, session_id: str, cfg: SessionConfig) -> None:
        with self._lock:
            self._configs[session_id] = cfg

    def status(self, session_id: str) -> Dict[str, str]:
        with self._lock:
            return dict(self._active.get(session_id, {}))

    def preview_webcam_png(self, webcam_index: int) -> Optional[bytes]:
        try:
            import cv2
        except Exception:
            return None
        cap = cv2.VideoCapture(int(webcam_index))
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return _to_png_bytes(img)

    def preview_serial_png(self, port: str, baud: int) -> Optional[bytes]:
        if not port:
            return None
        try:
            reader = SerialFrameReader(port=port, baud=int(baud))
            reader.open()
            try:
                raw = reader.read_frame(timeout_s=2.0)
            finally:
                reader.close()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((96, 96))
            img = Image.fromarray(arr, mode="L")
            return _to_png_bytes(img)
        except Exception:
            return None

    def _make_handler(self):
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    controller._handle(self)
                except Exception:
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"error")

            def log_message(self, format: str, *args) -> None:
                return

        return Handler

    def _handle(self, req: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(req.path)
        qs = parse_qs(parsed.query)
        path = parsed.path
        if path == "/status":
            session_id = (qs.get("session") or [""])[0]
            data = self.status(session_id)
            _send_json(req, data)
            return
        if path == "/start":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not source or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"})
                return
            self._start_record(session_id=session_id, source=source, class_name=class_name)
            _send_json(req, {"ok": "1"})
            return
        if path == "/stop":
            session_id = (qs.get("session") or [""])[0]
            self._stop_record(session_id=session_id)
            _send_json(req, {"ok": "1"})
            return
        _send_json(req, {"ok": "0", "error": "not found"}, status=404)

    def _start_record(self, session_id: str, source: str, class_name: str) -> None:
        with self._lock:
            cfg = self._configs.get(session_id)
            if cfg is None:
                raise RuntimeError("missing config")
            self._active[session_id] = {"recording": "1", "source": source, "class": class_name}

        t = threading.Thread(
            target=self._record_worker,
            args=(session_id, source, class_name),
            daemon=True,
        )
        t.start()

    def _stop_record(self, session_id: str) -> None:
        with self._lock:
            cur = self._active.get(session_id)
            if cur:
                cur["recording"] = "0"

    def _record_worker(self, session_id: str, source: str, class_name: str) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return
        interval = 1.0 / max(1.0, float(cfg.fps))
        if source == "device":
            self._record_serial(session_id, cfg, class_name, interval)
            return
        if source == "webcam":
            self._record_webcam(session_id, cfg, class_name, interval)
            return

    def _is_recording(self, session_id: str) -> bool:
        with self._lock:
            return self._active.get(session_id, {}).get("recording") == "1"

    def _record_serial(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        reader = SerialFrameReader(port=cfg.serial_port, baud=int(cfg.serial_baud))
        try:
            reader.open()
            while self._is_recording(session_id):
                raw = reader.read_frame(timeout_s=2.0)
                png = _raw96_to_png(raw, crop_box=cfg.crop_box)
                _save_png(cfg.dataset_root, class_name, png)
                time.sleep(interval)
        finally:
            try:
                reader.close()
            except Exception:
                pass

    def _record_webcam(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        try:
            import cv2
        except Exception:
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": "missing opencv-python"}
            return

        cap = cv2.VideoCapture(int(cfg.webcam_index))
        if not cap.isOpened():
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": "webcam open failed"}
            return
        try:
            while self._is_recording(session_id):
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(interval)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame).convert("L")
                if cfg.crop_box is not None:
                    x1, y1, x2, y2 = cfg.crop_box
                    img = img.crop((x1, y1, x2, y2))
                img = img.resize((96, 96))
                png = _to_png_bytes(img)
                _save_png(cfg.dataset_root, class_name, png)
                time.sleep(interval)
        finally:
            cap.release()


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _send_json(req: BaseHTTPRequestHandler, obj: Dict[str, str], status: int = 200) -> None:
    data = json.dumps(obj).encode("utf-8")
    req.send_response(status)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(data)))
    req.end_headers()
    req.wfile.write(data)


def _save_png(dataset_root: Path, class_name: str, png: bytes) -> Path:
    import uuid

    safe = sanitize_class_name(class_name)
    out_dir = dataset_root / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / (uuid.uuid4().hex + ".png")
    p.write_bytes(png)
    return p


def _raw96_to_png(raw: bytes, crop_box: Optional[Tuple[int, int, int, int]]) -> bytes:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((96, 96))
    img = Image.fromarray(arr, mode="L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        img = img.crop((x1, y1, x2, y2))
    img = img.resize((96, 96))
    return _to_png_bytes(img)


def _to_png_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def make_hold_button_html(label: str, start_url: str, stop_url: str) -> str:
    start_url_js = start_url.replace("'", "\\'")
    stop_url_js = stop_url.replace("'", "\\'")
    label_js = label.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div style="display:flex; flex-direction:column; gap:10px;">
  <button id="holdbtn" style="width:100%; padding:14px 12px; border-radius:10px; border:0; background:#0b5fff; color:white; font-weight:700; font-size:16px;">
    {label_js}
  </button>
  <div id="status" style="font-size:12px; color:rgba(0,0,0,.6);">按住开始录制，松开停止</div>
</div>
<script>
const btn = document.getElementById('holdbtn');
const status = document.getElementById('status');
let down = false;
async function doFetch(url) {{
  try {{ await fetch(url, {{method:'GET', mode:'no-cors'}}); }} catch (e) {{}}
}}
function onDown() {{
  if (down) return;
  down = true;
  btn.style.background = '#0647c6';
  status.textContent = '录制中... 松开停止';
  doFetch('{start_url_js}');
}}
function onUp() {{
  if (!down) return;
  down = false;
  btn.style.background = '#0b5fff';
  status.textContent = '已停止';
  doFetch('{stop_url_js}');
}}
btn.addEventListener('mousedown', onDown);
btn.addEventListener('mouseup', onUp);
btn.addEventListener('mouseleave', onUp);
btn.addEventListener('touchstart', function(e){{ e.preventDefault(); onDown(); }}, {{passive:false}});
btn.addEventListener('touchend', function(e){{ e.preventDefault(); onUp(); }}, {{passive:false}});
</script>
"""
