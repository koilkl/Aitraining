from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
import sys
import glob


HEADER = bytes([0xAA, 0x55, 0xAA])  # default frame sync header
DEFAULT_FRAME_SIDE = 96


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str


class SerialFrameReader:
    def __init__(self, port: str, baud: int, sync_header: bytes | str | None = None,
                 frame_side: int = DEFAULT_FRAME_SIDE, channels: int = 1) -> None:
        self._port = port
        self._baud = int(baud)
        self._header = parse_sync_header(sync_header)
        side = int(frame_side)
        if side < 8 or side > 512:
            raise ValueError("Invalid frame_side.")
        ch = int(channels)
        if ch not in (1, 3):
            raise ValueError("channels must be 1 (grayscale) or 3 (RGB).")
        self._frame_side = side
        self._channels = ch
        self._frame_size = int(side) * int(side) * ch
        self._ser = None
        self._buf = bytearray()

    def open(self) -> None:
        try:
            import serial
        except Exception as e:
            raise RuntimeError("pyserial is missing (serial module not available).") from e

        self._ser = serial.Serial(port=self._port, baudrate=self._baud, timeout=0.1)
        self._buf = bytearray()
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def read_frame(self, timeout_s: float = 3.0) -> bytes:
        if self._ser is None:
            raise RuntimeError("Serial not opened")
        start = time.time()
        header = self._header
        header_len = len(header)
        while time.time() - start < timeout_s:
            chunk = self._ser.read(4096)
            if chunk:
                self._buf.extend(chunk)
                # Keep at least 4× frame_size so we never discard a pending frame.
                # For 384×384×3 RGB that is ~1.7 MB — fine on a desktop.
                keep = int(self._frame_size) * 4
                if keep < 65536:
                    keep = 65536
                if len(self._buf) > keep * 2:
                    self._buf = self._buf[-keep:]
            else:
                time.sleep(0.004)
            while True:
                idx = self._buf.find(header)
                if idx < 0:
                    if len(self._buf) > header_len:
                        keep = max(0, header_len - 1)
                        self._buf = self._buf[-keep:] if keep else bytearray()
                    break
                after = idx + header_len
                need = after + self._frame_size
                if len(self._buf) < need:
                    if idx > 0:
                        del self._buf[:idx]
                    break
                if len(self._buf) >= need + header_len:
                    if self._buf[need : need + header_len] != header:
                        del self._buf[: idx + 1]
                        continue
                frame = bytes(self._buf[after:need])
                del self._buf[:need]
                print(f"[Serial] frame ok: {len(frame)} bytes (expecting {self._frame_size}), buf_remaining={len(self._buf)}")
                return frame
        raise TimeoutError(f"Timeout waiting for frame header (need {self._frame_size} bytes, buf has {len(self._buf)}, waited {timeout_s:.1f}s)")


def list_serial_ports() -> List[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except Exception:
        return _fallback_list_serial_ports()
    out: List[SerialPortInfo] = []
    for p in list_ports.comports():
        desc = getattr(p, "description", "") or ""
        out.append(SerialPortInfo(device=str(p.device), description=str(desc)))
    return out


def _fallback_list_serial_ports() -> List[SerialPortInfo]:
    patterns: List[str] = []
    if sys.platform == "darwin":
        patterns = ["/dev/cu.*", "/dev/tty.*"]
    elif sys.platform.startswith("linux"):
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyAMA*", "/dev/ttyS*"]
    else:
        patterns = []
    devices: List[str] = []
    for pat in patterns:
        devices.extend(glob.glob(pat))
    seen = set()
    out: List[SerialPortInfo] = []
    for dev in sorted(devices):
        if dev in seen:
            continue
        seen.add(dev)
        out.append(SerialPortInfo(device=str(dev), description=""))
    return out


def read_frame_png_from_serial(
    port: str,
    baud: int,
    sync_header: bytes | str | None = None,
    timeout_s: float = 3.0,
    frame_side: int = DEFAULT_FRAME_SIDE,
    channels: int = 1,
) -> bytes:
    reader = SerialFrameReader(port=port, baud=baud, sync_header=sync_header, frame_side=frame_side, channels=channels)
    reader.open()
    try:
        frame = reader.read_frame(timeout_s=timeout_s)
    finally:
        reader.close()
    side = int(frame_side)
    ch = int(channels)
    if ch == 3:
        arr = np.frombuffer(frame, dtype=np.uint8).reshape((side, side, 3))
        img = Image.fromarray(arr, mode="RGB")
    else:
        arr = np.frombuffer(frame, dtype=np.uint8).reshape((side, side))
        img = Image.fromarray(arr, mode="L")
    return _to_png_bytes(img)


def parse_sync_header(sync_header: bytes | str | None) -> bytes:
    if isinstance(sync_header, bytes):
        if not sync_header:
            raise ValueError("Sync header cannot be empty.")
        return sync_header
    raw = str(sync_header or "").strip()
    if not raw:
        return HEADER
    cleaned = raw.replace(",", " ").replace("0x", "").replace("0X", "")
    parts = [part.strip() for part in cleaned.split() if part.strip()]
    if not parts:
        return HEADER
    values = []
    for part in parts:
        if len(part) > 2:
            raise ValueError("Sync header bytes must be 1-2 hex digits.")
        try:
            value = int(part, 16)
        except Exception as e:
            raise ValueError("Sync header must use hex bytes like AA 55 AA.") from e
        if value < 0 or value > 255:
            raise ValueError("Sync header byte out of range.")
        values.append(value)
    if not values:
        raise ValueError("Sync header cannot be empty.")
    return bytes(values)


def _read_exact(ser, n: int, timeout_s: float) -> Optional[bytes]:
    start = time.time()
    data = bytearray()
    while len(data) < n and (time.time() - start) < timeout_s:
        chunk = ser.read(n - len(data))
        if chunk:
            data.extend(chunk)
    if len(data) != n:
        return None
    return bytes(data)


def _to_png_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()
