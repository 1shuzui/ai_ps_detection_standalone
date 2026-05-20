import hashlib
import math
from typing import List, Tuple

import cv2
import numpy as np


def _rng_from_key(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 55, 98))]
    ok, encoded = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        return image.copy()
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_perspective_tilt(image: np.ndarray, rng: np.random.Generator, max_shift_ratio: float = 0.035) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return image.copy()

    shift_x = max(2.0, w * max_shift_ratio)
    shift_y = max(2.0, h * max_shift_ratio * 0.8)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src.copy()

    for point in dst:
        point[0] = np.clip(point[0] + rng.uniform(-shift_x, shift_x), 0, w - 1)
        point[1] = np.clip(point[1] + rng.uniform(-shift_y, shift_y), 0, h - 1)

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_screen_moire(image: np.ndarray, rng: np.random.Generator, strength: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    theta = math.radians(float(rng.uniform(-35.0, 35.0)))
    period = float(rng.uniform(12.0, 26.0))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    coord_1 = xx * math.cos(theta) + yy * math.sin(theta)
    coord_2 = yy * math.cos(theta) - xx * math.sin(theta)
    wave = np.sin(coord_1 * (2.0 * math.pi / period) + phase)
    wave += 0.55 * np.sin(coord_2 * (2.0 * math.pi / (period * 1.7)) + phase / 2.0)

    overlay = wave[..., None] * (255.0 * strength)
    tinted = image.astype(np.float32) + overlay
    tinted += rng.uniform(-8.0, 8.0, size=(1, 1, 3)).astype(np.float32)
    return np.clip(tinted, 0, 255).astype(np.uint8)


def add_capture_chrome(image: np.ndarray, rng: np.random.Generator, include_nav_bar: bool = True) -> np.ndarray:
    h, w = image.shape[:2]
    status_h = max(24, int(h * 0.045))
    nav_h = max(34, int(h * 0.070)) if include_nav_bar else 0

    top_sample = image[: max(1, min(h, h // 12))]
    light_theme = float(np.mean(top_sample)) > 145.0
    bg_color = np.array([245, 245, 245], dtype=np.uint8) if light_theme else np.array([24, 24, 24], dtype=np.uint8)
    fg_color = (36, 36, 36) if light_theme else (245, 245, 245)
    separator = (210, 210, 210) if light_theme else (72, 72, 72)

    canvas = np.full((h + status_h + nav_h, w, 3), bg_color, dtype=np.uint8)
    canvas[status_h + nav_h :, :, :] = image

    time_text = f"{int(rng.integers(8, 24)):02d}:{int(rng.integers(0, 60)):02d}"
    font_scale = max(0.45, min(0.8, w / 1080.0 * 0.8))
    thickness = max(1, int(round(font_scale * 2)))
    time_org = (int(w * 0.06), int(status_h * 0.72))
    cv2.putText(canvas, time_text, time_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg_color, thickness, cv2.LINE_AA)

    battery_w = max(24, int(w * 0.062))
    battery_h = max(10, int(status_h * 0.34))
    battery_x2 = w - int(w * 0.05)
    battery_y1 = max(4, int(status_h * 0.28))
    battery_x1 = battery_x2 - battery_w
    battery_y2 = battery_y1 + battery_h
    cv2.rectangle(canvas, (battery_x1, battery_y1), (battery_x2, battery_y2), fg_color, 1)
    cap_x1 = battery_x2 + 1
    cap_y1 = battery_y1 + battery_h // 3
    cap_y2 = battery_y2 - battery_h // 3
    cv2.rectangle(canvas, (cap_x1, cap_y1), (cap_x1 + 3, cap_y2), fg_color, -1)
    level_x2 = battery_x1 + int(battery_w * float(rng.uniform(0.45, 0.95)))
    cv2.rectangle(canvas, (battery_x1 + 2, battery_y1 + 2), (max(battery_x1 + 3, level_x2), battery_y2 - 2), fg_color, -1)

    bar_x = battery_x1 - max(18, int(w * 0.02))
    for index in range(4):
        bar_h = max(4, int(status_h * (0.12 + 0.08 * index)))
        x = bar_x - index * 7
        cv2.rectangle(canvas, (x, battery_y2 - bar_h), (x + 4, battery_y2), fg_color, -1)

    if include_nav_bar:
        nav_y1 = status_h
        nav_y2 = status_h + nav_h
        arrow_x = int(w * 0.06)
        arrow_y = int((nav_y1 + nav_y2) / 2)
        arrow_len = max(10, int(w * 0.018))
        cv2.line(canvas, (arrow_x + arrow_len, arrow_y - arrow_len), (arrow_x, arrow_y), fg_color, 2, cv2.LINE_AA)
        cv2.line(canvas, (arrow_x, arrow_y), (arrow_x + arrow_len, arrow_y + arrow_len), fg_color, 2, cv2.LINE_AA)

        title_w = max(90, int(w * 0.22))
        title_h = max(10, int(nav_h * 0.18))
        title_x1 = (w - title_w) // 2
        title_y1 = arrow_y - title_h // 2
        cv2.rectangle(canvas, (title_x1, title_y1), (title_x1 + title_w, title_y1 + title_h), separator, -1)

        dot_x = w - int(w * 0.08)
        for offset in (-10, 0, 10):
            cv2.circle(canvas, (dot_x + offset, arrow_y), 2, fg_color, -1)

        cv2.line(canvas, (0, nav_y2 - 1), (w, nav_y2 - 1), separator, 1, cv2.LINE_AA)

    return canvas


def build_global_augmentations(image: np.ndarray, key: str) -> List[Tuple[str, np.ndarray]]:
    rng = _rng_from_key(f"global::{key}")
    variants: List[Tuple[str, np.ndarray]] = []

    tilted = apply_perspective_tilt(image, rng, max_shift_ratio=0.03)
    variants.append(("tilt", _jpeg_roundtrip(tilted, quality=int(rng.integers(82, 92)))))

    moire = apply_screen_moire(image, rng, strength=0.075)
    moire = cv2.GaussianBlur(moire, (3, 3), 0)
    variants.append(("moire", _jpeg_roundtrip(moire, quality=int(rng.integers(70, 86)))))

    chrome = add_capture_chrome(image, rng, include_nav_bar=True)
    variants.append(("top_chrome", chrome))

    chrome_capture = apply_screen_moire(chrome, rng, strength=0.06)
    chrome_capture = apply_perspective_tilt(chrome_capture, rng, max_shift_ratio=0.025)
    variants.append(("top_chrome_capture", _jpeg_roundtrip(chrome_capture, quality=int(rng.integers(72, 88)))))

    return variants


def build_roi_augmentations(roi_rgb: np.ndarray, key: str) -> List[Tuple[str, np.ndarray]]:
    h, w = roi_rgb.shape[:2]
    if h < 18 or w < 40:
        return []

    rng = _rng_from_key(f"roi::{key}")
    variants: List[Tuple[str, np.ndarray]] = []

    tilted = apply_perspective_tilt(roi_rgb, rng, max_shift_ratio=0.02)
    variants.append(("tilt", tilted))

    moire = apply_screen_moire(roi_rgb, rng, strength=0.05)
    moire = cv2.GaussianBlur(moire, (3, 3), 0)
    variants.append(("moire", _jpeg_roundtrip(moire, quality=int(rng.integers(76, 90)))))

    return variants
