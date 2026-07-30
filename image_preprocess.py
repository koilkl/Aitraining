from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


CLIP_MODE_SIGN = 0
CLIP_MODE_JUNCTION = 1
CLIP_MODE_BLUE_DIFF = 2  # B-G channel extraction for blue/purple signs

PREPROCESS_MODE_AUTO_BY_LABEL = "auto_by_label"
PREPROCESS_MODE_MANUAL_ROI = "manual_roi"
PREPROCESS_MODE_NONE = "none"
PREPROCESS_MODE_SIGN = "sign"
PREPROCESS_MODE_JUNCTION = "junction"
PREPROCESS_MODE_BLUE_DIFF = "blue_diff"  # B-G extraction for blue/purple signs

_SEARCH_LEFT_FRAC = 0.35
_SEARCH_RIGHT_FRAC = 0.65
_SEARCH_TOP_FRAC = 0.20
_SEARCH_BOTTOM_FRAC = 0.80
_FALLBACK_SIDE_FRAC = 0.30
_FALLBACK_CENTER_X_FRAC = 0.50
_FALLBACK_CENTER_Y_FRAC = 0.50
_CENTER_ROI_FRAC = 0.60  # fraction of image to keep from center for sign crop
_MIN_FOCUS_PIXELS = 18
_MIN_SIDE_FRAC = 0.50
_DARK_OFFSET = 10
_NEAR_BLACK_THRESH = 45  # max(R,G,B) below this → converted to pure white before auto-crop
_NEAR_WHITE_THRESH = 200  # R/G/B above this → kept as white (avoids stretch-to-black)
_BG_LUM_THRESH = 100  # max(R,G,B) above this → bright background → white
_BG_DIFF_MIN = 15    # |B-G| magnitude below this → low saturation → white
_CONTRAST_MIN_SPAN = 24
_EDGE_PERCENTILE = 0.90
_EDGE_MIN = 18
_EDGE_RELAX = 8
_SIGN_CENTER_BIAS_X_FRAC = 0.10
_SIGN_CENTER_BIAS_Y_FRAC = -0.03
_SIGN_PROJECTION_FRAC = 0.16
_SIGN_PRIOR_CENTER_X_FRAC = 0.50
_SIGN_PRIOR_CENTER_Y_FRAC = 0.50
_SIGN_PRIOR_SIGMA_X_FRAC = 0.18
_SIGN_PRIOR_SIGMA_Y_FRAC = 0.16
_SIGN_LOCAL_PEAK_RATIO = 0.68
_SIGN_LOCAL_SIDE_SCALE = 2.35
_SIGN_SUPPORT_PEAK_RATIO = 0.40
_SIGN_SUPPORT_SIDE_SCALE = 1.90
_SIGN_LOCAL_ASPECT_TARGET = 1.0
_SIGN_LOCAL_ASPECT_TOL = 0.75
_SIGN_FAR_SIDE_FRAC = 0.12
_SIGN_FAR_SIDE_BOOST = 1.16
_SIGN_CLOSE_SIDE_FRAC = 0.22
_SIGN_CLOSE_SIDE_BOOST = 1.42
_SIGN_BORDER_TOUCH_PX = 2
_SIGN_BORDER_SIDE_BOOST = 1.24
_SIGN_BORDER_SHIFT_FRAC = 0.16
_SIGN_PEAK_CENTER_BLEND = 0.68
_SIGN_CENTER_CLAMP_FRAC = 0.22
_JUNCTION_TOP_FRAC = 0.40
_JUNCTION_BOTTOM_FRAC = 0.75


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


def _center_bbox(h: int, w: int, frac: float = 0.60) -> Tuple[int, int, int, int]:
    """Simple center crop → sign is always in the middle of the frame."""
    side = int(min(h, w) * frac)
    cx, cy = w // 2, h // 2
    half = side // 2
    left = max(0, cx - half)
    top  = max(0, cy - half)
    right = min(w, left + side)
    bottom = min(h, top + side)
    return left, top, right, bottom


def _fallback_bbox(h: int, w: int) -> Tuple[int, int, int, int]:
    side = max(1, int(round(min(h, w) * _FALLBACK_SIDE_FRAC)))
    cx = int(round(w * _FALLBACK_CENTER_X_FRAC))
    cy = int(round(h * _FALLBACK_CENTER_Y_FRAC))
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))
    return int(left), int(top), int(left + side), int(top + side)


def _junction_bbox(h: int, w: int) -> Tuple[int, int, int, int]:
    top = max(0, min(h - 1, int(round(h * _JUNCTION_TOP_FRAC))))
    bottom = max(top + 1, min(h, int(round(h * _JUNCTION_BOTTOM_FRAC))))
    return 0, top, w, bottom


def _sign_search_window(h: int, w: int) -> Tuple[int, int, int, int]:
    left = max(0, min(w - 1, int(round(w * _SEARCH_LEFT_FRAC))))
    right = max(left + 1, min(w, int(round(w * _SEARCH_RIGHT_FRAC))))
    top = max(0, min(h - 1, int(round(h * _SEARCH_TOP_FRAC))))
    bottom = max(top + 1, min(h, int(round(h * _SEARCH_BOTTOM_FRAC))))
    return left, top, right, bottom


def _connected_bbox_from_seed(mask: np.ndarray, seed_x: int, seed_y: int) -> Optional[Tuple[int, int, int, int]]:
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return None
    sx = int(max(0, min(w - 1, seed_x)))
    sy = int(max(0, min(h - 1, seed_y)))
    if not mask[sy, sx]:
        return None
    visited = np.zeros_like(mask, dtype=bool)
    stack = [(sx, sy)]
    visited[sy, sx] = True
    min_x = max_x = sx
    min_y = max_y = sy
    while stack:
        px, py = stack.pop()
        min_x = min(min_x, px)
        max_x = max(max_x, px)
        min_y = min(min_y, py)
        max_y = max(max_y, py)
        if px > 0 and mask[py, px - 1] and not visited[py, px - 1]:
            visited[py, px - 1] = True
            stack.append((px - 1, py))
        if px + 1 < w and mask[py, px + 1] and not visited[py, px + 1]:
            visited[py, px + 1] = True
            stack.append((px + 1, py))
        if py > 0 and mask[py - 1, px] and not visited[py - 1, px]:
            visited[py - 1, px] = True
            stack.append((px, py - 1))
        if py + 1 < h and mask[py + 1, px] and not visited[py + 1, px]:
            visited[py + 1, px] = True
            stack.append((px, py + 1))
    return min_x, min_y, max_x, max_y


def _weighted_center_in_bbox(
    weight_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
    fallback_xy: Tuple[float, float],
) -> Tuple[float, float]:
    min_x, min_y, max_x, max_y = bbox
    x0 = max(0, min(weight_map.shape[1] - 1, int(min_x)))
    y0 = max(0, min(weight_map.shape[0] - 1, int(min_y)))
    x1 = max(x0 + 1, min(weight_map.shape[1], int(max_x) + 1))
    y1 = max(y0 + 1, min(weight_map.shape[0], int(max_y) + 1))
    window = np.asarray(weight_map[y0:y1, x0:x1], dtype=np.float32)
    if window.size == 0:
        return float(fallback_xy[0]), float(fallback_xy[1])
    weights = np.clip(window, 0.0, None)
    total = float(weights.sum())
    if total <= 1e-6:
        return float(fallback_xy[0]), float(fallback_xy[1])
    ys, xs = np.indices(window.shape, dtype=np.float32)
    cx = float((weights * xs).sum() / total) + float(x0)
    cy = float((weights * ys).sum() / total) + float(y0)
    return cx, cy


def _estimate_sign_center_and_side(
    target_map: np.ndarray, search_h: int, search_w: int, max_crop_side: int
) -> Optional[Tuple[float, float, int, float, float, float, Tuple[int, int, int, int]]]:
    if target_map.size == 0:
        return None

    kernel_size = max(3, int(round(min(search_h, search_w) * _SIGN_PROJECTION_FRAC)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32)

    score_map = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, target_map)
    score_map = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, score_map)
    ys = np.linspace(0.0, 1.0, search_h, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, search_w, dtype=np.float32)
    prior_x = np.exp(-0.5 * ((xs - _SIGN_PRIOR_CENTER_X_FRAC) / max(_SIGN_PRIOR_SIGMA_X_FRAC, 1e-6)) ** 2)
    prior_y = np.exp(-0.5 * ((ys - _SIGN_PRIOR_CENTER_Y_FRAC) / max(_SIGN_PRIOR_SIGMA_Y_FRAC, 1e-6)) ** 2)
    score_map = score_map * prior_y[:, None] * prior_x[None, :]
    peak_value = float(score_map.max(initial=0.0))
    if peak_value <= 0.0:
        return None

    local_mask = score_map >= (peak_value * _SIGN_LOCAL_PEAK_RATIO)
    h, w = local_mask.shape
    visited = np.zeros_like(local_mask, dtype=bool)
    best_score = -1.0
    best_span: Optional[Tuple[int, int]] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_peak: Optional[Tuple[int, int]] = None

    for y in range(h):
        for x in range(w):
            if not local_mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            weighted_sum = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            local_peak_value = -1.0
            local_peak_x = x
            local_peak_y = y

            while stack:
                px, py = stack.pop()
                value = float(score_map[py, px])
                weighted_sum += value
                weighted_x += value * float(px)
                weighted_y += value * float(py)
                if value > local_peak_value:
                    local_peak_value = value
                    local_peak_x = px
                    local_peak_y = py
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)

                if px > 0 and local_mask[py, px - 1] and not visited[py, px - 1]:
                    visited[py, px - 1] = True
                    stack.append((px - 1, py))
                if px + 1 < w and local_mask[py, px + 1] and not visited[py, px + 1]:
                    visited[py, px + 1] = True
                    stack.append((px + 1, py))
                if py > 0 and local_mask[py - 1, px] and not visited[py - 1, px]:
                    visited[py - 1, px] = True
                    stack.append((px, py - 1))
                if py + 1 < h and local_mask[py + 1, px] and not visited[py + 1, px]:
                    visited[py + 1, px] = True
                    stack.append((px, py + 1))

            if weighted_sum <= 0.0:
                continue

            span_w = int(max_x - min_x + 1)
            span_h = int(max_y - min_y + 1)
            aspect = float(span_w) / float(max(1, span_h))
            aspect_error = abs(aspect - _SIGN_LOCAL_ASPECT_TARGET)
            if aspect_error > _SIGN_LOCAL_ASPECT_TOL:
                continue

            aspect_score = max(0.2, 1.0 - aspect_error / max(_SIGN_LOCAL_ASPECT_TOL, 1e-6))
            area_score = min(1.0, float(span_w * span_h) / float(max(1, kernel_size * kernel_size)))
            center_x = (weighted_x / weighted_sum) / float(max(1, w - 1))
            center_y = (weighted_y / weighted_sum) / float(max(1, h - 1))
            center_score_x = float(np.exp(-0.5 * ((center_x - _SIGN_PRIOR_CENTER_X_FRAC) / max(_SIGN_PRIOR_SIGMA_X_FRAC, 1e-6)) ** 2))
            center_score_y = float(np.exp(-0.5 * ((center_y - _SIGN_PRIOR_CENTER_Y_FRAC) / max(_SIGN_PRIOR_SIGMA_Y_FRAC, 1e-6)) ** 2))
            center_score = 0.35 + 0.65 * center_score_x * center_score_y
            score = weighted_sum * aspect_score * (0.6 + 0.4 * area_score) * center_score

            if score > best_score:
                best_score = score
                best_span = (span_w, span_h)
                best_bbox = (min_x, min_y, max_x, max_y)
                best_peak = (local_peak_x, local_peak_y)

    if best_span is None or best_bbox is None or best_peak is None:
        return None

    span_w, span_h = best_span
    component_side_frac = float(max(span_w, span_h)) / float(max(1, min(search_h, search_w)))
    support_mask = score_map >= (peak_value * _SIGN_SUPPORT_PEAK_RATIO)
    support_bbox = _connected_bbox_from_seed(support_mask, best_peak[0], best_peak[1])
    if support_bbox is None:
        support_bbox = best_bbox
    sup_min_x, sup_min_y, sup_max_x, sup_max_y = support_bbox
    support_span = max(int(sup_max_x - sup_min_x + 1), int(sup_max_y - sup_min_y + 1))
    est_side = int(
        round(
            max(
                max(span_w, span_h) * _SIGN_LOCAL_SIDE_SCALE,
                float(support_span) * _SIGN_SUPPORT_SIDE_SCALE,
            )
        )
    )
    if component_side_frac <= _SIGN_FAR_SIDE_FRAC:
        est_side = int(round(float(est_side) * _SIGN_FAR_SIDE_BOOST))
    if component_side_frac >= _SIGN_CLOSE_SIDE_FRAC:
        est_side = int(round(float(est_side) * _SIGN_CLOSE_SIDE_BOOST))
    min_x, min_y, max_x, max_y = support_bbox
    touch_left = 1.0 if min_x <= _SIGN_BORDER_TOUCH_PX else 0.0
    touch_top = 1.0 if min_y <= _SIGN_BORDER_TOUCH_PX else 0.0
    touch_right = 1.0 if max_x >= search_w - 1 - _SIGN_BORDER_TOUCH_PX else 0.0
    touch_bottom = 1.0 if max_y >= search_h - 1 - _SIGN_BORDER_TOUCH_PX else 0.0
    if touch_left or touch_top or touch_right or touch_bottom:
        est_side = int(round(float(est_side) * _SIGN_BORDER_SIDE_BOOST))
    est_side = max(est_side, kernel_size)
    est_side = max(est_side, int(round(min(search_h, search_w) * _MIN_SIDE_FRAC)))
    est_side = min(est_side, max_crop_side)
    support_center = _weighted_center_in_bbox(
        target_map,
        support_bbox,
        fallback_xy=((float(sup_min_x + sup_max_x) * 0.5), (float(sup_min_y + sup_max_y) * 0.5)),
    )
    peak_cx = float(best_peak[0])
    peak_cy = float(best_peak[1])
    cx = (float(_SIGN_PEAK_CENTER_BLEND) * peak_cx) + ((1.0 - float(_SIGN_PEAK_CENTER_BLEND)) * float(support_center[0]))
    cy = (float(_SIGN_PEAK_CENTER_BLEND) * peak_cy) + ((1.0 - float(_SIGN_PEAK_CENTER_BLEND)) * float(support_center[1]))
    shift_x = (_SIGN_BORDER_SHIFT_FRAC * touch_right) - (_SIGN_BORDER_SHIFT_FRAC * touch_left)
    shift_y = (_SIGN_BORDER_SHIFT_FRAC * touch_bottom) - (_SIGN_BORDER_SHIFT_FRAC * touch_top)
    return cx, cy, est_side, component_side_frac, shift_x, shift_y, support_bbox


def _focus_bbox(gray: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = gray.shape
    left, top, right, bottom = _sign_search_window(h, w)
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
    dark_strength = np.clip(thr - region.astype(np.int16), 0, None).astype(np.float32)
    edge_strength = np.clip(edge.astype(np.int16) - edge_thr, 0, None).astype(np.float32)
    target_map = dark_strength * (edge_strength + 1.0)
    if int(mask.sum()) < _MIN_FOCUS_PIXELS:
        relax_thr = max(_EDGE_RELAX, edge_thr - 6)
        mask = (region <= p20) & (edge >= relax_thr)
        edge_strength = np.clip(edge.astype(np.int16) - relax_thr, 0, None).astype(np.float32)
        target_map = np.clip(p20 - region.astype(np.int16), 0, None).astype(np.float32) * (edge_strength + 1.0)
    if int(mask.sum()) < _MIN_FOCUS_PIXELS:
        return _fallback_bbox(h, w)

    estimated = _estimate_sign_center_and_side(target_map, region.shape[0], region.shape[1], min(h, w))
    if estimated is None:
        return _fallback_bbox(h, w)

    cx_local, cy_local, side, component_side_frac, border_shift_x, border_shift_y, support_bbox = estimated
    close_ratio = min(1.0, max(0.0, (component_side_frac - _SIGN_CLOSE_SIDE_FRAC) / max(1e-6, 0.40 - _SIGN_CLOSE_SIDE_FRAC)))
    bias_x = float(_SIGN_CENTER_BIAS_X_FRAC) * (1.0 - 0.75 * close_ratio)
    bias_y = float(_SIGN_CENTER_BIAS_Y_FRAC) * (1.0 - 0.50 * close_ratio)
    cx_local = cx_local + float(side) * (bias_x + border_shift_x)
    cy_local = cy_local + float(side) * (bias_y + border_shift_y)
    sup_min_x, sup_min_y, sup_max_x, sup_max_y = support_bbox
    support_span = float(max(sup_max_x - sup_min_x + 1, sup_max_y - sup_min_y + 1))
    support_center_x = float(sup_min_x + sup_max_x) * 0.5
    support_center_y = float(sup_min_y + sup_max_y) * 0.5
    center_limit = max(2.0, support_span * 0.5, float(side) * _SIGN_CENTER_CLAMP_FRAC)
    cx_local = float(np.clip(cx_local, support_center_x - center_limit, support_center_x + center_limit))
    cy_local = float(np.clip(cy_local, support_center_y - center_limit, support_center_y + center_limit))
    cx = float(left) + cx_local
    cy = float(top) + cy_local
    side = min(side, min(h, w))
    crop_left = max(0, min(w - side, int(round(cx - side * 0.5))))
    crop_top = max(0, min(h - side, int(round(cy - side * 0.5))))
    return int(crop_left), int(crop_top), int(crop_left + side), int(crop_top + side)


def _render_crop(crop: np.ndarray, out_size: int, color_mode: str, preserve_aspect: bool) -> np.ndarray:
    if color_mode == "grayscale":
        base = crop[:, :, 0] if crop.shape[-1] == 1 else crop[:, :, :3]
        img = Image.fromarray(base, mode="L" if crop.shape[-1] == 1 else "RGB").convert("L")
        if preserve_aspect:
            img = ImageOps.pad(img, (int(out_size), int(out_size)), method=Image.BILINEAR, color=0, centering=(0.5, 0.5))
        else:
            img = img.resize((int(out_size), int(out_size)), Image.BILINEAR)
        out = _contrast_stretch_u8(np.asarray(img, dtype=np.uint8)).astype(np.float32) / 255.0
        return np.expand_dims(out, axis=-1)

    rgb = crop[:, :, :3] if crop.shape[-1] >= 3 else np.repeat(crop[:, :, :1], 3, axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    if preserve_aspect:
        img = ImageOps.pad(img, (int(out_size), int(out_size)), method=Image.BILINEAR, color=(0, 0, 0), centering=(0.5, 0.5))
    else:
        img = img.resize((int(out_size), int(out_size)), Image.BILINEAR)
    out = np.asarray(img, dtype=np.uint8)
    return out.astype(np.float32) / 255.0


def class_clip_mode(class_name: str) -> int:
    name = str(class_name or "").upper()
    junction_tokens = ("CROSS", "CROSSROAD", "INTERSECTION", "T_JUNCTION", "TJUNCTION", "THREE_WAY", "FORK")
    if any(token in name for token in junction_tokens):
        return CLIP_MODE_JUNCTION
    return CLIP_MODE_SIGN


def normalize_manual_roi(roi: Any) -> Optional[Tuple[float, float, float, float]]:
    if roi is None:
        return None
    if isinstance(roi, dict):
        values = (
            roi.get("x1", roi.get("x")),
            roi.get("y1", roi.get("y")),
            roi.get("x2"),
            roi.get("y2"),
        )
        if values[2] is None and roi.get("w") is not None and values[0] is not None:
            values = (values[0], values[1], float(values[0]) + float(roi.get("w")), values[3])
        if values[3] is None and roi.get("h") is not None and values[1] is not None:
            values = (values[0], values[1], values[2], float(values[1]) + float(roi.get("h")))
    elif isinstance(roi, (list, tuple)) and len(roi) == 4:
        values = tuple(roi)
    else:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in values]
    except Exception:
        return None
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(x1 + 1e-6, min(1.0, x2))
    y2 = max(y1 + 1e-6, min(1.0, y2))
    return float(x1), float(y1), float(x2), float(y2)


def normalize_preprocess_mode(mode: Any) -> str:
    value = str(mode or PREPROCESS_MODE_AUTO_BY_LABEL).strip().lower()
    if value in {
        PREPROCESS_MODE_AUTO_BY_LABEL,
        PREPROCESS_MODE_MANUAL_ROI,
    }:
        return value
    return PREPROCESS_MODE_AUTO_BY_LABEL


def normalize_class_preprocess(raw: Any) -> Dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    mode = normalize_preprocess_mode(src.get("mode"))
    out: Dict[str, Any] = {"mode": mode}
    manual_roi = normalize_manual_roi(src.get("manual_roi"))
    if manual_roi is not None:
        out["manual_roi"] = list(manual_roi)
    # Preserve user-adjustable thresholds (clamped to valid range)
    bg_lum = int(src.get("bg_lum_thresh", 100))
    bg_diff = int(src.get("bg_diff_abs", _BG_DIFF_MIN))  # magnitude mode
    bg_dark = int(src.get("bg_dark_thresh", 0))
    out["bg_lum_thresh"] = max(50, min(255, bg_lum))
    out["bg_diff_abs"] = max(0, min(255, bg_diff))  # |B-G| magnitude
    out["bg_dark_thresh"] = max(0, min(100, bg_dark))

    return out


def normalize_class_preprocess_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name:
            continue
        out[name] = normalize_class_preprocess(value)
    return out


def normalize_sample_preprocess_map(raw: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for class_name, value in raw.items():
        class_key = str(class_name or "").strip()
        if not class_key or not isinstance(value, dict):
            continue
        sample_map: Dict[str, Dict[str, Any]] = {}
        for sample_name, sample_cfg in value.items():
            filename = str(sample_name or "").strip()
            if not filename:
                continue
            sample_map[filename] = normalize_class_preprocess(sample_cfg)
        if sample_map:
            out[class_key] = sample_map
    return out


def resolve_preprocess_config(
    label_name: str,
    preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL,
    manual_roi: Any = None,
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if class_preprocess:
        item = class_preprocess.get(str(label_name or ""))
        if isinstance(item, dict):
            normalized = normalize_class_preprocess(item)
            item_mode = str(normalized.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
            if item_mode == PREPROCESS_MODE_MANUAL_ROI:
                return {
                    "mode": item_mode,
                    "manual_roi": normalize_manual_roi(normalized.get("manual_roi")),
                }
    mode = normalize_preprocess_mode(preprocess_mode)
    if mode == PREPROCESS_MODE_AUTO_BY_LABEL:
        return {"mode": PREPROCESS_MODE_AUTO_BY_LABEL, "manual_roi": None}
    return {"mode": mode, "manual_roi": normalize_manual_roi(manual_roi)}


def manual_roi_to_pixels(h: int, w: int, roi: Any) -> Optional[Tuple[int, int, int, int]]:
    norm = normalize_manual_roi(roi)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    px1 = max(0, min(w - 1, int(round(x1 * w))))
    py1 = max(0, min(h - 1, int(round(y1 * h))))
    px2 = max(px1 + 1, min(w, int(round(x2 * w))))
    py2 = max(py1 + 1, min(h, int(round(y2 * h))))
    return px1, py1, px2, py2


def preprocess_array(arr: np.ndarray, out_size: int, color_mode: str = "grayscale", clip_mode: int = CLIP_MODE_SIGN) -> np.ndarray:
    src = _to_uint8_image(arr)
    if src.ndim == 2:
        src = src[:, :, None]
    gray = src[:, :, 0] if src.shape[-1] == 1 else np.asarray(Image.fromarray(src[:, :, :3], mode="RGB").convert("L"), dtype=np.uint8)
    if int(clip_mode) == CLIP_MODE_JUNCTION:
        x1, y1, x2, y2 = _junction_bbox(gray.shape[0], gray.shape[1])
        preserve_aspect = True
    else:
        # Use G channel for shadow search — best SNR in green-biased lighting.
        # Dark sign absorbs green → low G.  White background reflects → high G.
        x1, y1, x2, y2 = _focus_bbox(g.astype(np.uint8))
        preserve_aspect = False
    crop = src[y1:y2, x1:x2]
    if crop.size == 0:
        crop = src
    return _render_crop(crop, out_size=out_size, color_mode=color_mode, preserve_aspect=preserve_aspect)


def _find_bg_roi(rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find the largest blue/purple region in an RGB image and return its bbox.

    Pipeline:
    1. Convert near-black (all channels < 30) and near-white (all > 200)
       pixels to pure white so dark/bright background don't confuse detection.
    2. Compute B-G difference on the cleaned image (white → 0, blue sign → +).
    3. Gaussian blur → contrast stretch → binary mask (>80).
    4. Morphological clean-up (erode 1×, dilate 2×).
    5. Find largest contour → square crop box with 20 % padding.

    Returns None when no blue/purple region is found.
    """
    try:
        import cv2
    except Exception:
        return None  # OpenCV not available → fall back to full frame

    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None

    # Step 1 — Convert near-black and near-white pixels to pure white first.
    # Dark / shadow areas and bright-white background can both confuse the
    # auto-crop detector.  Replacing both with pure white makes the sign
    # the only non-background region in the frame.
    r_raw = rgb[:, :, 0].astype(np.int16)
    g_raw = rgb[:, :, 1].astype(np.int16)
    b_raw = rgb[:, :, 2].astype(np.int16)
    near_black = np.maximum(np.maximum(r_raw, g_raw), b_raw) < _NEAR_BLACK_THRESH
    near_white = (r_raw > _NEAR_WHITE_THRESH) & (g_raw > _NEAR_WHITE_THRESH) & (b_raw > _NEAR_WHITE_THRESH)
    background = near_black | near_white

    clean_rgb = rgb.copy()
    clean_rgb[background] = [255, 255, 255]

    # Step 2 — B-G difference on the cleaned image.
    # White pixels → B-G = 0 (neutral).  Blue/purple sign → B > G → bright.
    r = clean_rgb[:, :, 0].astype(np.int16)
    g = clean_rgb[:, :, 1].astype(np.int16)
    b = clean_rgb[:, :, 2].astype(np.int16)
    bg_gray = (b - g).astype(np.float32)
    bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)

    bg_min = float(bg_blur.min())
    bg_max = float(bg_blur.max())
    bg_span = bg_max - bg_min
    if bg_span < 20:                            # no meaningful contrast
        return None

    stretched = ((bg_blur - bg_min) / bg_span * 255).astype(np.uint8)
    mask = (stretched > 80).astype(np.uint8) * 255   # bright = sign, lower = sensitive

    if mask.sum() < 16:
        return None

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours → largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < 16.0:                                  # too small
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # Expand to square with 20 % padding
    side = max(bw, bh)
    pad = max(1, int(side * 0.20))
    side += pad * 2
    cx, cy = x + bw // 2, y + bh // 2
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def preprocess_blue_diff_array(arr: np.ndarray, out_size: int, color_mode: str = "grayscale",
                               roi: Optional[Tuple[int, int, int, int]] = None,
                               bg_dark_thresh: int = 0,
                               bg_lum_thresh: int = 100,
                               return_crop_box: bool = False,
                               fast_mode: bool = False):
    """Simplified pipeline for all-black signs on white background.

    Uses raw G channel (best SNR in green-biased lighting).
    Sign = dark pixels (G between dark_thresh and lum_thresh).
    No colour processing needed — black is black in any light.
    """
    src = _to_uint8_image(arr)
    if src.ndim == 2:
        src = src[:, :, None]
    if src.shape[-1] < 3:
        src = np.repeat(src[:, :, :1], 3, axis=2)

    # Use raw G channel as luminance — best SNR in green lighting.
    # Black sign absorbs all light → low G.  White paper reflects → high G.
    gray = src[:, :, 1].astype(np.uint8)

    # Simple dark/lum mask: keep only mid-brightness pixels (the sign).
    # Too dark (below dark_thresh) → shadow/noise → white.
    # Too bright (above lum_thresh) → paper/background → white.
    is_sign = (gray > bg_dark_thresh) & (gray < bg_lum_thresh)
    sign_pct = is_sign.mean() * 100
    too_dark_pct = (gray <= bg_dark_thresh).mean() * 100
    too_bright_pct = (gray >= bg_lum_thresh).mean() * 100
    print(f"[MASK] sign={sign_pct:.1f}% dark={too_dark_pct:.1f}% bright={too_bright_pct:.1f}%  (G>{bg_dark_thresh} & G<{bg_lum_thresh})")
    gray[~is_sign] = 255

    # Crop: shadow search (live preview) or center crop (batch/cache)
    h_orig, w_orig = gray.shape[:2]
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(gray.shape[1] - 1, x1))
        y1 = max(0, min(gray.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(gray.shape[1], x2))
        y2 = max(y1 + 1, min(gray.shape[0], y2))
    elif fast_mode:
        # Fast: simple center crop (for batch cache rebuild)
        x1, y1, x2, y2 = _center_bbox(h_orig, w_orig, frac=0.60)
    else:
        # Live: _focus_bbox dark-object + edge detection in center window
        x1, y1, x2, y2 = _focus_bbox(gray)

    # Save normalized crop box for ROI overlay display
    crop_norm = (x1 / w_orig, y1 / h_orig, x2 / w_orig, y2 / h_orig)

    gray = gray[y1:y2, x1:x2]

    # Resize
    img = Image.fromarray(gray, mode="L")
    img = img.resize((int(out_size), int(out_size)), Image.BILINEAR)

    # Contrast stretch — matches C++ side exactly
    out_arr = np.asarray(img, dtype=np.uint8)
    out = _contrast_stretch_u8(out_arr)
    result = out.astype(np.float32) / 255.0
    image = np.expand_dims(result, axis=-1)  # (96,96) → (96,96,1)
    if return_crop_box:
        return image, crop_norm
    return image


def preprocess_manual_roi_array(arr: np.ndarray, out_size: int, color_mode: str = "grayscale", manual_roi: Any = None) -> np.ndarray:
    src = _to_uint8_image(arr)
    if src.ndim == 2:
        src = src[:, :, None]
    bbox = manual_roi_to_pixels(src.shape[0], src.shape[1], manual_roi)
    if bbox is None:
        return _render_crop(src, out_size=out_size, color_mode=color_mode, preserve_aspect=True)
    x1, y1, x2, y2 = bbox
    crop = src[y1:y2, x1:x2]
    if crop.size == 0:
        crop = src
    return _render_crop(crop, out_size=out_size, color_mode=color_mode, preserve_aspect=True)


def preprocess_for_label(
    arr: np.ndarray,
    out_size: int,
    color_mode: str = "grayscale",
    label_name: str = "",
    preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL,
    manual_roi: Any = None,
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
) -> np.ndarray:
    resolved = resolve_preprocess_config(
        label_name=label_name,
        preprocess_mode=preprocess_mode,
        manual_roi=manual_roi,
        class_preprocess=class_preprocess,
    )
    mode = str(resolved.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
    roi: Any = None
    if mode == PREPROCESS_MODE_MANUAL_ROI:
        resolved_roi = resolved.get("manual_roi")
        src = _to_uint8_image(arr)
        roi = manual_roi_to_pixels(src.shape[0], src.shape[1], resolved_roi)
    # Always B-G; mode only controls how the ROI is chosen
    return preprocess_blue_diff_array(arr, out_size=out_size, color_mode=color_mode, roi=roi)


def prepare_inference_inputs(
    arr: np.ndarray,
    out_size: int,
    color_mode: str = "grayscale",
    preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL,
    manual_roi: Any = None,
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_dark_thresh: int = 0,
    bg_lum_thresh: int = 100,
) -> Dict[str, np.ndarray]:
    mode = normalize_preprocess_mode(preprocess_mode)
    roi: Any = None
    if mode == PREPROCESS_MODE_MANUAL_ROI:
        src = _to_uint8_image(arr)
        roi = manual_roi_to_pixels(src.shape[0], src.shape[1], manual_roi)
    return {"default": preprocess_blue_diff_array(arr, out_size=out_size, color_mode=color_mode, roi=roi,
                                                  bg_dark_thresh=int(bg_dark_thresh),
                                                  bg_lum_thresh=int(bg_lum_thresh))}


def focus_and_enhance_array(arr: np.ndarray, out_size: int, color_mode: str = "grayscale") -> np.ndarray:
    return preprocess_array(arr, out_size=out_size, color_mode=color_mode, clip_mode=CLIP_MODE_SIGN)
