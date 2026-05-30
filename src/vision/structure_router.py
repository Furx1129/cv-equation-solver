from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.normalization import normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image
from src.vision.structure_2d import _crop_binary, _is_fraction_line, _line_candidates


AUTO_ROUTE_CATEGORIES = (
    "printed_basic",
    "printed_decimal_negative",
    "printed_2d_layout",
    "handwritten_basic",
    "calculus",
)


@dataclass(frozen=True)
class ImageStructureFeatures:
    width: int
    height: int
    aspect_ratio: float
    foreground_components: int
    left_tall_components: int
    long_horizontal_lines: int
    fraction_like_lines: int
    stacked_component_pairs: int
    edge_roughness: float
    foreground_fill_ratio: float

    @property
    def is_extremely_flat_row(self) -> bool:
        return (
            self.aspect_ratio > 5.0
            and self.fraction_like_lines == 0
            and self.long_horizontal_lines == 0
            and self.stacked_component_pairs <= 1
        )

    @property
    def is_single_symbol_like(self) -> bool:
        return (
            self.foreground_components <= 3
            and self.aspect_ratio < 2.2
            and self.fraction_like_lines == 0
            and self.long_horizontal_lines <= 1
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "foreground_components": self.foreground_components,
            "left_tall_components": self.left_tall_components,
            "long_horizontal_lines": self.long_horizontal_lines,
            "fraction_like_lines": self.fraction_like_lines,
            "stacked_component_pairs": self.stacked_component_pairs,
            "edge_roughness": round(self.edge_roughness, 4),
            "foreground_fill_ratio": round(self.foreground_fill_ratio, 4),
        }


@dataclass(frozen=True)
class FormulaStructureAnalysis:
    binary: np.ndarray
    crop_bbox: tuple[int, int, int, int]
    features: ImageStructureFeatures
    route_hint: str
    route_confidence: float
    debug_reason: str
    fallback_categories: tuple[str, ...]


def analyze_formula_structure(image_path: str | Path) -> FormulaStructureAnalysis:
    image = read_image(image_path)
    normalized = normalize_formula_image(image)
    preprocessed = preprocess_image(
        normalized.image,
        options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
    )
    binary = _crop_binary(preprocessed.binary)
    features = extract_structure_features(binary)
    route_hint, confidence, reason, fallbacks = choose_route(features)
    return FormulaStructureAnalysis(
        binary=binary,
        crop_bbox=normalized.crop_bbox,
        features=features,
        route_hint=route_hint,
        route_confidence=confidence,
        debug_reason=reason,
        fallback_categories=fallbacks,
    )


def extract_image_structure_features(image_path: str | Path) -> ImageStructureFeatures:
    return analyze_formula_structure(image_path).features


def extract_structure_features(binary: np.ndarray) -> ImageStructureFeatures:
    height, width = binary.shape[:2]
    foreground = binary < 255
    fg_count = int(foreground.sum())
    fill = fg_count / max(1, width * height)

    component_boxes = _component_boxes(binary)
    lines = _line_candidates(binary)
    long_lines = [line for line in lines if line.bbox[2] >= max(28, width * 0.12)]
    fraction_lines = [line for line in lines if _is_fraction_line(binary, line)]
    edges = cv2.Canny(binary, 50, 150)
    edge_roughness = float((edges > 0).sum() / max(1, fg_count))

    return ImageStructureFeatures(
        width=width,
        height=height,
        aspect_ratio=width / max(1, height),
        foreground_components=len(component_boxes),
        left_tall_components=_count_left_tall_components(component_boxes, width, height),
        long_horizontal_lines=len(long_lines),
        fraction_like_lines=len(fraction_lines),
        stacked_component_pairs=_count_stacked_pairs(component_boxes),
        edge_roughness=edge_roughness,
        foreground_fill_ratio=fill,
    )


def choose_route(features: ImageStructureFeatures) -> tuple[str, float, str, tuple[str, ...]]:
    if features.foreground_components <= 1 and features.aspect_ratio < 2.2:
        return (
            "handwritten_basic",
            0.90,
            "single connected foreground component, treated as a single handwritten symbol",
            ("printed_basic", "printed_decimal_negative", "printed_2d_layout"),
        )

    if _looks_like_handwritten_expression(features):
        return (
            "handwritten_basic",
            0.78,
            "rough low-fill foreground with a few handwritten symbol components",
            ("printed_basic", "printed_decimal_negative", "printed_2d_layout"),
        )

    if _looks_like_calculus(features):
        return (
            "calculus",
            0.86,
            "calculus-like structure: tall left marker, fraction lines, or rough multi-part layout",
            ("printed_2d_layout", "printed_basic", "printed_decimal_negative"),
        )

    if _looks_like_2d_layout(features):
        return (
            "printed_2d_layout",
            0.84,
            "two-dimensional structure detected from horizontal lines or stacked components",
            ("calculus", "printed_basic", "printed_decimal_negative"),
        )

    if features.is_single_symbol_like:
        return (
            "handwritten_basic",
            0.82,
            "compact foreground with no strong two-dimensional layout markers",
            ("printed_2d_layout", "printed_basic", "printed_decimal_negative"),
        )

    return (
        "printed_basic",
        0.76 if features.is_extremely_flat_row else 0.68,
        "flat or mostly linear expression without strong two-dimensional markers",
        ("printed_decimal_negative", "printed_2d_layout", "handwritten_basic"),
    )


def _looks_like_calculus(features: ImageStructureFeatures) -> bool:
    if features.is_single_symbol_like:
        return False
    if (
        features.left_tall_components > 0
        and 0.20 < features.edge_roughness < 0.38
        and features.foreground_fill_ratio < 0.20
    ):
        return True
    if (
        0.24 < features.edge_roughness < 0.38
        and features.foreground_fill_ratio < 0.18
        and features.foreground_components >= 5
        and (features.fraction_like_lines > 0 or features.stacked_component_pairs >= 3)
    ):
        return True
    if (
        features.fraction_like_lines > 0
        and features.aspect_ratio < 3.4
        and features.left_tall_components > 0
        and features.stacked_component_pairs >= 2
    ):
        return True
    return False


def _looks_like_handwritten_expression(features: ImageStructureFeatures) -> bool:
    return (
        features.foreground_components <= 5
        and features.edge_roughness >= 0.38
        and features.foreground_fill_ratio < 0.12
        and features.stacked_component_pairs <= 1
    )


def _looks_like_2d_layout(features: ImageStructureFeatures) -> bool:
    if features.is_extremely_flat_row:
        return False
    if (
        features.aspect_ratio >= 3.6
        and features.fraction_like_lines == 0
        and features.long_horizontal_lines <= 1
        and features.stacked_component_pairs <= 1
    ):
        return False
    if features.fraction_like_lines > 0 or features.long_horizontal_lines > 0:
        return True
    return features.stacked_component_pairs > 1 and features.aspect_ratio < 3.8


def _component_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    foreground = (binary < 255).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(5, int(binary.shape[0] * binary.shape[1] * 0.0002))
    for label in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area >= min_area and w >= 2 and h >= 2:
            boxes.append((x, y, w, h))
    return boxes


def _count_stacked_pairs(boxes: list[tuple[int, int, int, int]]) -> int:
    count = 0
    for index, first in enumerate(boxes):
        fx, fy, fw, fh = first
        for second in boxes[index + 1 :]:
            sx, sy, sw, sh = second
            vertical_gap = max(sy - (fy + fh), fy - (sy + sh), 0)
            if vertical_gap <= 2:
                continue
            overlap = max(0, min(fx + fw, sx + sw) - max(fx, sx))
            if overlap / max(1, min(fw, sw)) >= 0.25:
                count += 1
    return count


def _count_left_tall_components(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> int:
    return sum(1 for x, _, w, h in boxes if x < width * 0.30 and h >= height * 0.55 and w <= width * 0.35)
