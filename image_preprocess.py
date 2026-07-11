from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


_SEARCH_LEFT_FRAC = 0.18
_SEARCH_RIGHT_FRAC = 0.82
_SEARCH_TOP_FRAC = 0.22
_SEARCH_BOTTOM_FRAC = 0.95
_FALLBACK_SIDE_FRAC = 0.64
_FALLBACK_CENTER_Y_FRAC = 0.60
_MIN_FOCUS_PIXELS = 18
_MIN_SIDE_FRAC = 0.34
_EXPAND_SCALE = 2.6
_DARK_OFFSET = 18
_CONTRAST_MIN_SPAN = 24
_EDGE_PERCENTILE = 0.80
_EDGE_MIN = 12
_EDGE_RELAX = 8


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    src = np.asarray(arr)
    if src.dtype == np.uint8:
        return src
    src = src.astype(np.float32)
    if float(np.nanmax(src)) <= 1.5:
        src = src * 255.0
    src = np.clip(src, 0.0, 255.0)
    return src.astype(np.uint8)


def _percentile_from_hist(hist: np.ndarray, total: int, q: float) -> int:
    if total <= 0:
        return 0
    rank = int(max(0, min(total - 1, round((total - 1) * float(q)))))
    cdf = np.cumsum(hist, dtype=np.int64)
    idx = int(np.searchsorted(cdf, rank + 1, side="left"))
    return max(0, min(255, idx))


def _fallback_bbox(h: int, w: int) -> Tuple[int, int, int, int]:
    side = max(1, int(round(min(h, w) * _FALLBACK_SIDE_FRAC)))
    cx = w // 2
    cy = int(round(h * _FALLBACK_CENTER_Y_FRAC))
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))
    return int(left), int(top), int(left + side), int(top + side)


def _focus_bbox(gray: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = gray.shape
    left = max(0, min(w - 1, int(round(w * _SEARCH_LEFT_FRAC))))
    right = max(left + 1, min(w, int(round(w * _SEARCH_RIGHT_FRAC))))
    top = max(0, min(h - 1, int(round(h * _SEARCH_TOP_FRAC))))
    bottom = max(top + 1, min(h, int(round(h * _SEARCH_BOTTOM_FRAC))))

    region = gray[top:bottom, left:right]
    if region.size == 0:
        return _fallback_bbox(h, w)

    hist = np.bincount(region.reshape(-1), minlength=256)
    total = int(region.size)
    p20 = _percentile_from_hist(hist, total, 0.20)
    p35 = _percentile_from_hist(hist, total, 0.35)
    thr = min(p35, p20 + _DARK_OFFSET)

    edge = _edge_map_u8(region)
    edge_hist = np.bincount(edge.reshape(-1), minlength=256)
    edge_thr = max(_EDGE_MIN, _percentile_from_hist(edge_hist, total, _EDGE_PERCENTILE))

    mask = (region <= thr) & (edge >= edge_thr)
    if int(mask.sum()) < _MIN_FOCUS_PIXELS:
        relax_thr = max(_EDGE_RELAX, edge_thr - 6)
        mask = (region <= p20) & (edge >= relax_thr)
    if int(mask.sum()) < _MIN_FOCUS_PIXELS:
        return _fallback_bbox(h, w)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return _fallback_bbox(h, w)

    x1 = left + int(xs.min())
    x2 = left + int(xs.max()) + 1
    y1 = top + int(ys.min())
    y2 = top + int(ys.max()) + 1

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = int(round(max(bw, bh) * _EXPAND_SCALE))
    side = max(side, int(round(min(h, w) * _MIN_SIDE_FRAC)))
    side = min(side, min(h, w))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    crop_left = max(0, min(w - side, cx - side // 2))
    crop_top = max(0, min(h - side, cy - side // 2))
    return int(crop_left), int(crop_top), int(crop_left + side), int(crop_top + side)


def _contrast_stretch_u8(arr: np.ndarray) -> np.ndarray:
    src = np.asarray(arr, dtype=np.uint8)
    low = int(src.min())
    high = int(src.max())
    span = high - low
    if span < _CONTRAST_MIN_SPAN:
        return src
    out = (src.astype(np.int32) - low) * 255 + span // 2
    out = out // span
    return np.clip(out, 0, 255).astype(np.uint8)


def _edge_map_u8(gray: np.ndarray) -> np.ndarray:
    src = np.asarray(gray, dtype=np.int16)
    gx = np.zeros_like(src, dtype=np.int16)
    gy = np.zeros_like(src, dtype=np.int16)
    gx[:, 1:-1] = np.abs(src[:, 2:] - src[:, :-2])
    gx[:, 0] = np.abs(src[:, 1] - src[:, 0])
    gx[:, -1] = np.abs(src[:, -1] - src[:, -2])
    gy[1:-1, :] = np.abs(src[2:, :] - src[:-2, :])
    gy[0, :] = np.abs(src[1, :] - src[0, :])
    gy[-1, :] = np.abs(src[-1, :] - src[-2, :])
    edge = (gx + gy) // 2
    return np.clip(edge, 0, 255).astype(np.uint8)


def focus_and_enhance_array(arr: np.ndarray, out_size: int, color_mode: str = "grayscale") -> np.ndarray:
    src = _to_uint8_image(arr)
    if src.ndim == 2:
        src = src[:, :, None]
    if src.shape[-1] == 1:
        gray = src[:, :, 0]
    else:
        gray = np.asarray(Image.fromarray(src[:, :, :3], mode="RGB").convert("L"), dtype=np.uint8)

    x1, y1, x2, y2 = _focus_bbox(gray)
    crop = src[y1:y2, x1:x2]
    if crop.size == 0:
        crop = src

    if color_mode == "grayscale":
        img = Image.fromarray(crop[:, :, 0] if crop.shape[-1] == 1 else crop[:, :, :3], mode="L" if crop.shape[-1] == 1 else "RGB").convert("L")
        img = img.resize((int(out_size), int(out_size)), Image.BILINEAR)
        out = _contrast_stretch_u8(np.asarray(img, dtype=np.uint8)).astype(np.float32) / 255.0
        return np.expand_dims(out, axis=-1)

    img = Image.fromarray(crop[:, :, :3] if crop.shape[-1] >= 3 else np.repeat(crop[:, :, :1], 3, axis=2), mode="RGB")
    img = img.resize((int(out_size), int(out_size)), Image.BILINEAR)
    out = np.asarray(img, dtype=np.uint8)
    stretch_gray = _contrast_stretch_u8(np.asarray(img.convert("L"), dtype=np.uint8))
    if out.ndim == 3 and out.shape[-1] == 3:
        low = int(stretch_gray.min())
        high = int(stretch_gray.max())
        span = high - low
        if span >= _CONTRAST_MIN_SPAN:
            out_i32 = (out.astype(np.int32) - low) * 255 + span // 2
            out = np.clip(out_i32 // span, 0, 255).astype(np.uint8)
    return out.astype(np.float32) / 255.0
