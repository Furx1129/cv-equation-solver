from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    image: np.ndarray
    crop_bbox: tuple[int, int, int, int]
    scale: float
    output_size: tuple[int, int]


def foreground_bbox(binary_image: np.ndarray, foreground_threshold: int = 250) -> tuple[int, int, int, int]:
    if binary_image.ndim != 2:
        raise ValueError("foreground_bbox expects a single-channel image")
    mask = binary_image < foreground_threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        h, w = binary_image.shape
        return (0, 0, w, h)
    x1 = int(cols[0])
    x2 = int(cols[-1])
    y1 = int(rows[0])
    y2 = int(rows[-1])
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def normalize_formula_image(
    image: np.ndarray,
    target_height: int = 256,
    margin: int = 24,
    foreground_threshold: int = 250,
) -> NormalizationResult:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y, w, h = foreground_bbox(binary, foreground_threshold=foreground_threshold)
    img_h, img_w = gray.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(img_w, x + w + margin)
    y1 = min(img_h, y + h + margin)
    cropped = gray[y0:y1, x0:x1]

    if cropped.size == 0:
        raise ValueError("normalization produced an empty crop")

    scale = target_height / cropped.shape[0]
    target_width = max(1, int(round(cropped.shape[1] * scale)))
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, target_width + 2 * margin), 255, dtype=np.uint8)
    canvas[:, margin : margin + target_width] = resized
    _, normalized = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return NormalizationResult(
        image=normalized,
        crop_bbox=(x0, y0, x1 - x0, y1 - y0),
        scale=scale,
        output_size=(normalized.shape[1], normalized.shape[0]),
    )
