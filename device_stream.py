from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class DeviceFrame:
    width: int
    height: int
    mode: str
    png_bytes: bytes


class DeviceStream:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_frame: Optional[DeviceFrame] = None
        self._last_error: Optional[str] = None
        self._url: Optional[str] = None

    def start(self, url: str) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._url == url:
                return
            self.stop()
            self._url = url
            self._stop.clear()
            t = threading.Thread(target=self._run, args=(url,), daemon=True)
            self._thread = t
            t.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self._thread = None
            self._url = None

    def last_frame(self) -> Optional[DeviceFrame]:
        with self._lock:
            return self._last_frame

    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._last_error = msg

    def _set_frame(self, frame: DeviceFrame) -> None:
        with self._lock:
            self._last_frame = frame
            self._last_error = None

    def _run(self, url: str) -> None:
        try:
            import websocket
        except Exception as e:
            self._set_error(f"Missing dependency websocket-client: {e}")
            return

        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(url, timeout=5)
                try:
                    ws.settimeout(2)
                    try:
                        ws.send("tm-connected")
                    except Exception:
                        pass
                    while not self._stop.is_set():
                        msg = ws.recv()
                        if not msg:
                            continue
                        if isinstance(msg, bytes):
                            continue
                        raw = base64.b64decode(msg)
                        if len(raw) == 96 * 96:
                            arr = np.frombuffer(raw, dtype=np.uint8).reshape((96, 96))
                            img = Image.fromarray(arr, mode="L")
                            buf = _to_png_bytes(img)
                            self._set_frame(DeviceFrame(width=96, height=96, mode="L", png_bytes=buf))
                finally:
                    try:
                        ws.close()
                    except Exception:
                        pass
            except Exception as e:
                self._set_error(str(e))
                time.sleep(0.5)


def _to_png_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()
