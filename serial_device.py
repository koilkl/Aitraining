from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image


HEADER = bytes([0xAA, 0x55, 0xAA])
FRAME_W = 96
FRAME_H = 96
FRAME_SIZE = FRAME_W * FRAME_H


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str


class SerialFrameReader:
    def __init__(self, port: str, baud: int) -> None:
        self._port = port
        self._baud = int(baud)
        self._ser = None

    def open(self) -> None:
        import serial

        self._ser = serial.Serial(port=self._port, baudrate=self._baud, timeout=0.1)
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
        header_pos = 0
        while time.time() - start < timeout_s:
            chunk = self._ser.read(4096)
            if not chunk:
                continue
            for b in chunk:
                if b == HEADER[header_pos]:
                    header_pos += 1
                    if header_pos == len(HEADER):
                        header_pos = 0
                        frame = _read_exact(self._ser, FRAME_SIZE, timeout_s=max(0.2, timeout_s - (time.time() - start)))
                        if frame is None:
                            raise TimeoutError("Timeout while reading frame bytes")
                        return frame
                else:
                    header_pos = 0
        raise TimeoutError("Timeout waiting for frame header")


def list_serial_ports() -> List[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    out: List[SerialPortInfo] = []
    for p in list_ports.comports():
        desc = getattr(p, "description", "") or ""
        out.append(SerialPortInfo(device=str(p.device), description=str(desc)))
    return out


def read_frame_png_from_serial(
    port: str,
    baud: int,
    timeout_s: float = 3.0,
) -> bytes:
    reader = SerialFrameReader(port=port, baud=baud)
    reader.open()
    try:
        frame = reader.read_frame(timeout_s=timeout_s)
    finally:
        reader.close()
    arr = np.frombuffer(frame, dtype=np.uint8).reshape((FRAME_H, FRAME_W))
    img = Image.fromarray(arr, mode="L")
    return _to_png_bytes(img)


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
