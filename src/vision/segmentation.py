from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    bbox: tuple[int, int, int, int]
    image: np.ndarray


def _foreground_mask(binary_image: np.ndarray) -> np.ndarray:
    if binary_image.ndim != 2:
        raise ValueError("segmentation expects a single-channel binary image")
    return binary_image < 255


def segment_characters(
    binary_image: np.ndarray,
    min_width: int = 1,
    min_height: int = 1,
    min_gap: int = 1,
    pad: int = 2,
) -> list[Segment]:
    foreground = _foreground_mask(binary_image)
    cols = np.where(foreground.any(axis=0))[0]
    if cols.size == 0:
        raise ValueError("no foreground pixels found; cannot segment expression")

    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    previous = int(cols[0])

    for col in map(int, cols[1:]):
        if col - previous > min_gap:
            runs.append((start, previous))
            start = col
        previous = col
    runs.append((start, previous))

    height, width = binary_image.shape
    segments: list[Segment] = []
    for x1, x2 in runs:
        region = foreground[:, x1 : x2 + 1]
        rows = np.where(region.any(axis=1))[0]
        if rows.size == 0:
            continue
        y1 = max(0, int(rows[0]) - pad)
        y2 = min(height - 1, int(rows[-1]) + pad)
        x1p = max(0, x1 - pad)
        x2p = min(width - 1, x2 + pad)
        w = x2p - x1p + 1
        h = y2 - y1 + 1
        if w < min_width or h < min_height:
            continue
        roi = binary_image[y1 : y2 + 1, x1p : x2p + 1]
        segments.append(Segment(bbox=(x1p, y1, w, h), image=roi))

    if not segments:
        raise ValueError("foreground was found, but no valid character segments were produced")
    return segments
