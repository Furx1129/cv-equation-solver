from __future__ import annotations

import cv2
import numpy as np

from src.vision.types import SymbolRegion


def segment_symbols(
    binary_image: np.ndarray,
    source: str = "connected_component",
    min_area: int = 8,
    min_width: int = 2,
    min_height: int = 2,
    merge_distance: int = 3,
) -> list[SymbolRegion]:
    if binary_image.ndim != 2:
        raise ValueError("segment_symbols expects a single-channel image")
    foreground = (binary_image < 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    boxes: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or w < min_width or h < min_height:
            continue
        boxes.append((x, y, w, h, area))

    boxes = _merge_nearby_boxes(sorted(boxes, key=lambda b: (b[0], b[1])), merge_distance=merge_distance)
    regions = [_box_to_region(binary_image, box, source=source) for box in boxes]
    return sorted(regions, key=lambda r: (r.bbox[0], r.bbox[1]))


def split_wide_symbol(region: SymbolRegion, max_aspect_ratio: float = 2.6) -> list[SymbolRegion]:
    if region.aspect_ratio <= max_aspect_ratio:
        return [region]
    image = region.image
    foreground = image < 255
    col_counts = foreground.sum(axis=0)
    threshold = max(1, int(col_counts.max() * 0.12))
    valleys = np.where(col_counts <= threshold)[0]
    if valleys.size == 0:
        return [region]
    split_col = int(valleys[len(valleys) // 2])
    if split_col <= 1 or split_col >= image.shape[1] - 2:
        return [region]
    left = image[:, :split_col]
    right = image[:, split_col:]
    x, y, _, h = region.bbox
    return [
        _image_to_region(left, (x, y, split_col, h), source=f"{region.source}_split"),
        _image_to_region(right, (x + split_col, y, image.shape[1] - split_col, h), source=f"{region.source}_split"),
    ]


def split_wide_symbols(regions: list[SymbolRegion], max_aspect_ratio: float = 2.6) -> list[SymbolRegion]:
    result: list[SymbolRegion] = []
    for region in regions:
        result.extend(split_wide_symbol(region, max_aspect_ratio=max_aspect_ratio))
    return sorted(result, key=lambda r: (r.bbox[0], r.bbox[1]))


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    merge_distance: int,
) -> list[tuple[int, int, int, int, int]]:
    merged: list[tuple[int, int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        last = merged[-1]
        lx, ly, lw, lh, la = last
        x, y, w, h, area = box
        gap = x - (lx + lw)
        vertical_overlap = min(ly + lh, y + h) - max(ly, y)
        if gap <= merge_distance and vertical_overlap > 0:
            x0 = min(lx, x)
            y0 = min(ly, y)
            x1 = max(lx + lw, x + w)
            y1 = max(ly + lh, y + h)
            merged[-1] = (x0, y0, x1 - x0, y1 - y0, la + area)
        else:
            merged.append(box)
    return merged


def _box_to_region(binary_image: np.ndarray, box: tuple[int, int, int, int, int], source: str) -> SymbolRegion:
    x, y, w, h, area = box
    return _image_to_region(binary_image[y : y + h, x : x + w], (x, y, w, h), source=source, area=area)


def _image_to_region(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    source: str,
    area: int | None = None,
) -> SymbolRegion:
    x, y, w, h = bbox
    if area is None:
        area = int((image < 255).sum())
    aspect_ratio = w / max(1, h)
    return SymbolRegion(
        bbox=(x, y, w, h),
        image=image,
        source=source,
        area=area,
        aspect_ratio=aspect_ratio,
        baseline_position="middle",
    )
