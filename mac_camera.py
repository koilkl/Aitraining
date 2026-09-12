"""Open a macOS camera by AVCaptureDevice uniqueID.

cv2.VideoCapture only opens cameras by positional index, but the index→name
mapping the UI lists (AVCaptureDevice ``devicesWithMediaType`` order) is not
stable — the system order flips whenever virtual cameras (Iriun etc.) connect
or disconnect, so a stored index points at a DIFFERENT camera than the name
says.  This module captures via AVCaptureSession against a specific uniqueID,
so the camera the user picked IS the camera that streams.

Exposes ``open_macos_camera_by_unique_id(unique_id)`` → ``MacCameraCapture``
or None.  ``MacCameraCapture`` mimics the cv2.VideoCapture surface the
callers use: ``isOpened()``, ``read()`` → ``(ok, BGR ndarray)``,
``release()``.
"""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Optional

import numpy as np
import objc
from AVFoundation import (
    AVCaptureDevice,
    AVCaptureDeviceInput,
    AVCaptureSession,
    AVCaptureVideoDataOutput,
)
from CoreMedia import CMSampleBufferGetImageBuffer
from Foundation import NSObject
from Quartz import CoreVideo as CV

# libdispatch bindings are not always installed (pyobjc 12 split them into a
# separate package) — call dispatch_queue_create directly from libSystem.
_dispatch = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_dispatch.dispatch_queue_create.restype = ctypes.c_void_p
_dispatch.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]


def _dispatch_queue_create(label: str):
    raw = _dispatch.dispatch_queue_create(label.encode("utf-8"), None)
    # Must be wrapped as an ObjC proxy — passing the bare pointer to the
    # bridge segfaults when AVFoundation dispatches the first sample buffer.
    return objc.objc_object(c_void_p=raw)


class _SampleDelegate(NSObject):
    """AVCaptureVideoDataOutput sample-buffer delegate."""

    def initWithCallback_(self, callback):
        self = objc.super(_SampleDelegate, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sampleBuffer, connection):
        try:
            self._callback(sampleBuffer)
        except Exception:
            pass


def _pointer_to_bytes(addr, n: int):
    # pyobjc 12 returns an objc.varlist for void* returns; as_buffer(n) is the
    # fast C-level copy.  Older pyobjc returned an int (ctypes) or a buffer.
    if hasattr(addr, "as_buffer"):
        buf = addr.as_buffer(int(n))
        return buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
    if hasattr(addr, "tobytes"):
        return addr.tobytes()
    if isinstance(addr, int):
        return ctypes.string_at(addr, int(n))
    return b"".join(addr[0:int(n)])


def _sample_buffer_to_bgr(sample_buffer) -> Optional[np.ndarray]:
    image_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
    if image_buffer is None:
        return None
    CV.CVPixelBufferLockBaseAddress(image_buffer, 0)
    try:
        addr = CV.CVPixelBufferGetBaseAddress(image_buffer)
        w = int(CV.CVPixelBufferGetWidth(image_buffer))
        h = int(CV.CVPixelBufferGetHeight(image_buffer))
        stride = int(CV.CVPixelBufferGetBytesPerRow(image_buffer))
        if not addr or w <= 0 or h <= 0 or stride < w * 4:
            return None
        raw = _pointer_to_bytes(addr, stride * h)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride)
        # BGRA row, possibly padded to stride — cut to width, drop alpha.
        return np.ascontiguousarray(arr[:, : w * 4].reshape(h, w, 4)[:, :, :3])
    finally:
        CV.CVPixelBufferUnlockBaseAddress(image_buffer, 0)


class MacCameraCapture:
    """Minimal cv2.VideoCapture-compatible wrapper around AVCaptureSession."""

    def __init__(self, device) -> None:
        self._device = device
        self._session = None
        self._output = None
        self._delegate = None
        self._cond = threading.Condition()
        self._latest_bgr: Optional[np.ndarray] = None
        self._closed = False

    def _start(self) -> bool:
        session = AVCaptureSession.alloc().init()
        try:
            device_input = AVCaptureDeviceInput.deviceInputWithDevice_error_(self._device, None)
        except Exception:
            return False
        if isinstance(device_input, tuple):
            device_input = device_input[0] if device_input else None
        if device_input is None:
            return False
        session.addInput_(device_input)
        output = AVCaptureVideoDataOutput.alloc().init()
        output.setVideoSettings_({CV.kCVPixelBufferPixelFormatTypeKey: CV.kCVPixelFormatType_32BGRA})
        output.setAlwaysDiscardsLateVideoFrames_(True)
        delegate = _SampleDelegate.alloc().initWithCallback_(self._on_frame)
        queue = _dispatch_queue_create("com.aitraining.maccamera")
        output.setSampleBufferDelegate_queue_(delegate, queue)
        session.addOutput_(output)
        session.startRunning()
        self._session = session
        self._output = output
        self._delegate = delegate
        return True

    def _on_frame(self, sample_buffer) -> None:
        bgr = _sample_buffer_to_bgr(sample_buffer)
        if bgr is None:
            return
        with self._cond:
            self._latest_bgr = bgr
            self._cond.notify_all()

    def isOpened(self) -> bool:
        return self._session is not None and not self._closed

    def read(self, timeout_s: float = 3.0):
        deadline = time.time() + max(0.1, float(timeout_s))
        with self._cond:
            while self._latest_bgr is None and time.time() < deadline and not self._closed:
                self._cond.wait(0.1)
            frame = self._latest_bgr
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._closed = True
        session, output, delegate = self._session, self._output, self._delegate
        self._session = None
        self._output = None
        self._delegate = None
        try:
            if session is not None:
                session.stopRunning()
        except Exception:
            pass
        try:
            if output is not None and delegate is not None:
                output.setSampleBufferDelegate_queue_(None, None)
        except Exception:
            pass
        try:
            if delegate is not None:
                delegate._callback = None
        except Exception:
            pass


def open_macos_camera_by_unique_id(unique_id: str, timeout_s: float = 4.0) -> Optional[MacCameraCapture]:
    """Open the EXACT camera for ``unique_id`` and wait for the first frame."""
    try:
        device = AVCaptureDevice.deviceWithUniqueID_(str(unique_id))
    except Exception:
        return None
    if device is None:
        return None
    cap = MacCameraCapture(device)
    try:
        if not cap._start():
            cap.release()
            return None
        ok, frame = cap.read(timeout_s=timeout_s)
        if not ok or frame is None:
            cap.release()
            return None
        return cap
    except Exception:
        try:
            cap.release()
        except Exception:
            pass
        return None
