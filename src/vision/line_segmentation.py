from __future__ import annotations

import numpy as np

from src.vision.types import LineRegion


def segment_lines(
    binary_image: np.ndarray,
    min_height: int = 8,
    min_gap: int = 2,
    pad: int = 3,
) -> list[LineRegion]:
    if binary_image.ndim != 2:
        raise ValueError("segment_lines expects a single-channel image")
    foreground = binary_image < 255
    rows = np.where(foreground.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("no foreground pixels found; cannot segment lines")

    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    previous = int(rows[0])
    for row in map(int, rows[1:]):
        if row - previous > min_gap:
            runs.append((start, previous))
            start = row
        previous = row
    runs.append((start, previous))

    height, width = binary_image.shape
    lines: list[LineRegion] = []
    for y1, y2 in runs:
        if y2 - y1 + 1 < min_height:
            continue
        y0 = max(0, y1 - pad)
        y3 = min(height - 1, y2 + pad)
        region = foreground[y0 : y3 + 1, :]
        cols = np.where(region.any(axis=0))[0]
        if cols.size == 0:
            continue
        x0 = max(0, int(cols[0]) - pad)
        x1 = min(width - 1, int(cols[-1]) + pad)
        image = binary_image[y0 : y3 + 1, x0 : x1 + 1]
        baseline = _estimate_baseline(image)
        lines.append(LineRegion(bbox=(x0, y0, x1 - x0 + 1, y3 - y0 + 1), image=image, baseline_y=baseline))

    if not lines:
        raise ValueError("foreground was found, but no valid line regions were produced")
    return lines


def _estimate_baseline(line_image: np.ndarray) -> int:
    foreground = line_image < 255
    projection = foreground.sum(axis=1)
    if projection.size == 0:
        return line_image.shape[0] // 2
    return int(np.argmax(projection))
