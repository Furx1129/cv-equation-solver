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
    vertical_symmetry: float = 0.0
    component_height_variance: float = 0.0
    horizontal_run_count: int = 0
    left_right_density_ratio: float = 1.0
    multi_scale_edge_ratio: float = 1.0
    top_alignment_score: float = 0.0

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

    @property
    def is_vertically_symmetric(self) -> bool:
        return self.vertical_symmetry > 0.7

    @property
    def is_handwritten_texture(self) -> bool:
        return self.edge_roughness > 0.22 or (
            self.multi_scale_edge_ratio > 1.4 and self.component_height_variance > 0.12
        )

    @property
    def has_integral_structure(self) -> bool:
        return self.left_right_density_ratio > 1.5 and self.left_tall_components > 0

    @property
    def is_grid_aligned(self) -> bool:
        return self.top_alignment_score > 0.3

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
            "vertical_symmetry": round(self.vertical_symmetry, 4),
            "component_height_variance": round(self.component_height_variance, 4),
            "horizontal_run_count": self.horizontal_run_count,
            "left_right_density_ratio": round(self.left_right_density_ratio, 4),
            "multi_scale_edge_ratio": round(self.multi_scale_edge_ratio, 4),
            "top_alignment_score": round(self.top_alignment_score, 4),
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
    edges_low = cv2.Canny(binary, 30, 90)
    edges_high = cv2.Canny(binary, 50, 150)
    edge_roughness = float((edges_high > 0).sum() / max(1, fg_count))
    low_edge_count = int((edges_low > 0).sum())
    high_edge_count = int((edges_high > 0).sum())

    top_half_fg = int(foreground[: height // 2, :].sum()) if height >= 2 else 0
    bottom_half_fg = int(foreground[height // 2 :, :].sum()) if height >= 2 else 0
    vertical_symmetry = 1.0 - abs(top_half_fg - bottom_half_fg) / max(1, top_half_fg + bottom_half_fg)

    heights = [h for _, _, _, h in component_boxes]
    if len(heights) >= 2:
        mean_h = sum(heights) / len(heights)
        std_h = (sum((h - mean_h) ** 2 for h in heights) / len(heights)) ** 0.5
        height_variance = std_h / max(1, mean_h)
    else:
        height_variance = 0.0

    horizontal_proj = foreground.sum(axis=1).astype(np.float32)
    noise_floor = float(np.mean(horizontal_proj)) * 0.3 if horizontal_proj.size > 0 else 0.0
    in_run = False
    run_count = 0
    for val in horizontal_proj:
        if val > noise_floor and not in_run:
            run_count += 1
            in_run = True
        elif val <= noise_floor:
            in_run = False
    horizontal_run_count = run_count

    left_fg = int(foreground[:, : width // 2].sum()) if width >= 2 else 0
    right_fg = int(foreground[:, width // 2 :].sum()) if width >= 2 else 0
    left_right_density_ratio = left_fg / max(1, right_fg)

    multi_scale_edge_ratio = low_edge_count / max(1, high_edge_count)

    top_ys = [y for y, _, _, _ in component_boxes]
    if len(top_ys) >= 2:
        mean_top = sum(top_ys) / len(top_ys)
        std_top = (sum((y - mean_top) ** 2 for y in top_ys) / len(top_ys)) ** 0.5
        top_alignment = 1.0 / (1.0 + std_top)
    else:
        top_alignment = 0.0

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
        vertical_symmetry=vertical_symmetry,
        component_height_variance=height_variance,
        horizontal_run_count=horizontal_run_count,
        left_right_density_ratio=left_right_density_ratio,
        multi_scale_edge_ratio=multi_scale_edge_ratio,
        top_alignment_score=top_alignment,
    )


def _score_printed_basic(features: ImageStructureFeatures) -> float:
    s = 0.50
    if features.is_grid_aligned:
        s += 0.20
    if features.edge_roughness < 0.15:
        s += 0.15
    if features.fraction_like_lines == 0:
        s += 0.10
    if features.component_height_variance < 0.15:
        s += 0.10
    if features.stacked_component_pairs <= 1:
        s += 0.05
    if features.is_extremely_flat_row:
        s += 0.05
    if features.fraction_like_lines > 0:
        s -= 0.15
    return max(0.0, s)


def _score_printed_decimal_negative(features: ImageStructureFeatures) -> float:
    return _score_printed_basic(features)


def _score_printed_2d_layout(features: ImageStructureFeatures) -> float:
    s = 0.40
    if features.fraction_like_lines > 0:
        s += 0.25
    if features.stacked_component_pairs > 1:
        s += 0.15
    if features.long_horizontal_lines > 0:
        s += 0.15
    if features.horizontal_run_count >= 3:
        s += 0.10
    if features.aspect_ratio < 3.8:
        s += 0.05
    if features.left_tall_components > 0:
        s += 0.05
    if features.is_extremely_flat_row:
        s -= 0.30
    if features.is_handwritten_texture:
        s -= 0.25
    return max(0.0, s)


def _score_handwritten_basic(features: ImageStructureFeatures) -> float:
    s = 0.42
    if features.is_single_symbol_like:
        s += 0.30
    if features.is_handwritten_texture:
        s += 0.25
    if features.edge_roughness > 0.25:
        s += 0.12
    if features.foreground_components <= 5:
        s += 0.15
    if features.foreground_fill_ratio < 0.15:
        s += 0.10
    if not features.is_grid_aligned:
        s += 0.10
    if features.fraction_like_lines > 0:
        s -= 0.25
    if features.stacked_component_pairs > 1:
        s -= 0.15
    return max(0.0, s)
def _score_calculus(features: ImageStructureFeatures) -> float:
    s = 0.35
    if features.has_integral_structure:
        s += 0.30
    if features.fraction_like_lines > 0:
        s += 0.20
    if features.stacked_component_pairs >= 2:
        s += 0.10
    if features.stacked_component_pairs >= 8:
        s += 0.08
    if features.edge_roughness > 0.20:
        s += 0.10
    if features.left_tall_components > 0:
        s += 0.10
    if 1.5 < features.aspect_ratio < 4.0:
        s += 0.05
    if features.is_single_symbol_like:
        s -= 0.30
    if features.is_extremely_flat_row:
        s -= 0.25
    return max(0.0, s)


_SCORING_FUNCTIONS = {
    "printed_basic": _score_printed_basic,
    "printed_decimal_negative": _score_printed_decimal_negative,
    "printed_2d_layout": _score_printed_2d_layout,
    "handwritten_basic": _score_handwritten_basic,
    "calculus": _score_calculus,
}


def choose_route(features: ImageStructureFeatures) -> tuple[str, float, str, tuple[str, ...]]:
    scores = {cat: fn(features) for cat, fn in _SCORING_FUNCTIONS.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_category, top_score = ranked[0]
    second_category, second_score = ranked[1]
    margin = top_score - second_score
    fallbacks = tuple(cat for cat, _ in ranked[1:])
    if margin >= 0.12:
        return top_category, top_score, f"clear margin {margin:.2f} over {second_category}", fallbacks
    else:
        return top_category, top_score * 0.7, f"narrow margin {margin:.2f} vs {second_category}", fallbacks


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
