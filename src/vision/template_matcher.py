from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


TemplateFiles = dict[str, tuple[str, ...]]


def read_grayscale_image(path: str | Path) -> np.ndarray | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = np.fromfile(str(file_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


TEMPLATE_FILES: TemplateFiles = {
    "0": ("tem_0.jpg", "tem_0_gen.png"),
    "1": ("tem_1.jpg", "tem_1_gen.png"),
    "2": ("tem_2.jpg", "tem_2_gen.png"),
    "3": ("tem_3.jpg", "tem_3_gen.png"),
    "4": ("tem_4.jpg", "tem_4_gen.png"),
    "5": ("tem_5.jpg", "tem_5_gen.png"),
    "6": ("tem_6.jpg", "tem_6_gen.png"),
    "7": ("tem_7.jpg", "tem_7_gen.png"),
    "8": ("tem_8.jpg", "tem_8_gen.png"),
    "9": ("tem_9.jpg", "tem_9_gen.png"),
    "+": ("tem_plus.jpg", "tem_plus_gen.png"),
    "-": ("tem_minus.jpg", "tem_minus_gen.png"),
    "x": ("tem_times.jpg", "tem_mul_x.png"),
    "÷": ("tem_divide.jpg", "tem_divide_symbol.png"),
    "(": ("tem_left.jpg", "tem_left_gen.png"),
    ")": ("tem_right.jpg", "tem_right_gen.png"),
    "=": ("tem_equal.png",),
    ".": ("tem_dot.png",),
    "y": ("tem_var_y.png",),
    "^": ("tem_caret.png",),
    "√": ("tem_sqrt.png",),
    "∫": ("tem_integral.png",),
    "d": ("tem_d.png",),
    "∂": ("tem_partial.png",),
    "→": ("tem_arrow.png",),
    "∞": ("tem_infinity.png",),
    "l": ("tem_l.png",),
    "i": ("tem_i.png",),
    "m": ("tem_m.png",),
    "s": ("tem_s.png",),
    "n": ("tem_n.png",),
    "c": ("tem_c.png",),
    "o": ("tem_o.png",),
    "t": ("tem_t.png",),
    "a": ("tem_a.png",),
    "e": ("tem_e.png",),
    "p": ("tem_p.png",),
    "g": ("tem_g.png",),
    "r": ("tem_r.png",),
    "q": ("tem_q.png",),
    "lim": ("tem_lim.png",),
    "dx": ("tem_dx.png",),
    "d/dx": ("tem_d_over_dx.png",),
}


@dataclass(frozen=True)
class TemplateMatch:
    label: str
    score: float
    candidates: dict[str, float]


def _trim_foreground(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("template matching expects single-channel images")
    mask = image < 255
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image
    return image[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def normalize_symbol_image(image: np.ndarray, size: int = 64, margin: int = 8) -> np.ndarray:
    trimmed = _trim_foreground(image)
    h, w = trimmed.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("cannot normalize an empty symbol image")

    inner = size - 2 * margin
    scale = min(inner / w, inner / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(trimmed, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), 255, dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    _, normalized = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)
    return normalized


def extract_geometry_features(binary: np.ndarray) -> dict[str, float]:
    """Extract scale-invariant geometry features from a binary symbol image."""
    mask = binary < 128
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    empty = {
        "num_holes": 0.0,
        "num_components": 0.0,
        "aspect_ratio": 1.0,
        "centroid_x": 0.5,
        "centroid_y": 0.5,
        "density_top": 0.0,
        "density_bottom": 0.0,
        "top_bottom_ratio": 0.0,
        "left_right_ratio": 0.0,
        "fill_ratio": 0.0,
        "horizontal_score": 0.0,
        "vertical_score": 0.0,
        "diag_pos_score": 0.0,
        "diag_neg_score": 0.0,
    }
    if rows.size == 0 or cols.size == 0:
        return empty

    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 2 or bbox_w < 2:
        return empty

    cropped_mask = mask[y1:y2, x1:x2]
    foreground = cropped_mask.astype(np.uint8)

    # Count enclosed background components. This is more stable for generated
    # handwritten strokes than contour hierarchy after downscaling.
    background = (foreground == 0).astype(np.uint8)
    bg_count, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    border_labels = set(bg_labels[0, :])
    border_labels.update(bg_labels[-1, :])
    border_labels.update(bg_labels[:, 0])
    border_labels.update(bg_labels[:, -1])
    min_hole_area = max(8, int(round(bbox_w * bbox_h * 0.01)))
    num_holes = sum(
        1
        for index in range(1, bg_count)
        if index not in border_labels and bg_stats[index, cv2.CC_STAT_AREA] >= min_hole_area
    )

    _, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    num_components = 0
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] >= 4:
            num_components += 1

    total_fg = max(int(np.sum(cropped_mask)), 1)
    third = max(bbox_h // 3, 1)
    top_region = cropped_mask[:third, :]
    bottom_region = cropped_mask[bbox_h - third :, :]
    density_top = float(np.sum(top_region)) / total_fg
    density_bottom = float(np.sum(bottom_region)) / total_fg
    half_h = max(bbox_h // 2, 1)
    half_w = max(bbox_w // 2, 1)
    top = float(np.sum(cropped_mask[:half_h, :]))
    bottom = float(np.sum(cropped_mask[half_h:, :]))
    left = float(np.sum(cropped_mask[:, :half_w]))
    right = float(np.sum(cropped_mask[:, half_w:]))

    moments = cv2.moments(foreground)
    if moments["m00"]:
        centroid_x = float(moments["m10"] / moments["m00"] / max(bbox_w, 1))
        centroid_y = float(moments["m01"] / moments["m00"] / max(bbox_h, 1))
    else:
        centroid_x = 0.5
        centroid_y = 0.5

    line_scores = _line_direction_scores(foreground)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    fill_ratio = float(np.sum(cropped_mask)) / max(bbox_w * bbox_h, 1)

    return {
        "num_holes": float(num_holes),
        "num_components": float(num_components),
        "aspect_ratio": round(aspect_ratio, 4),
        "centroid_x": round(centroid_x, 4),
        "centroid_y": round(centroid_y, 4),
        "density_top": round(density_top, 4),
        "density_bottom": round(density_bottom, 4),
        "top_bottom_ratio": round(top / (bottom + 1.0), 4),
        "left_right_ratio": round(left / (right + 1.0), 4),
        "fill_ratio": round(fill_ratio, 4),
        "horizontal_score": round(line_scores["horizontal"], 4),
        "vertical_score": round(line_scores["vertical"], 4),
        "diag_pos_score": round(line_scores["diag_pos"], 4),
        "diag_neg_score": round(line_scores["diag_neg"], 4),
    }


def _line_direction_scores(foreground: np.ndarray) -> dict[str, float]:
    height, width = foreground.shape
    if height < 4 or width < 4:
        return {"horizontal": 0.0, "vertical": 0.0, "diag_pos": 0.0, "diag_neg": 0.0}
    edges = cv2.Canny((foreground * 255).astype(np.uint8), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(8, min(width, height) // 5),
        maxLineGap=5,
    )
    scores = {"horizontal": 0.0, "vertical": 0.0, "diag_pos": 0.0, "diag_neg": 0.0}
    if lines is None:
        return scores
    scale = max(width + height, 1)
    for [[x1, y1, x2, y2]] in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length == 0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) < 15 or abs(abs(angle) - 180) < 15:
            scores["horizontal"] += length
        elif abs(abs(angle) - 90) < 15:
            scores["vertical"] += length
        elif angle > 0:
            scores["diag_pos"] += length
        else:
            scores["diag_neg"] += length
    return {key: value / scale for key, value in scores.items()}


def precompute_template_features(
    templates: dict[str, list[np.ndarray]],
) -> dict[str, list[dict[str, float]]]:
    """Precompute geometry features for every template image."""
    features: dict[str, list[dict[str, float]]] = {}
    for label, images in templates.items():
        features[label] = [extract_geometry_features(img) for img in images]
    return features


def load_templates(
    template_dir: str | Path,
    size: int = 64,
    include_labels: set[str] | frozenset[str] | None = None,
    exclude_filenames: set[str] | frozenset[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    base = Path(template_dir)
    templates: dict[str, list[np.ndarray]] = {}
    missing: list[str] = []
    for label, filenames in TEMPLATE_FILES.items():
        if include_labels is not None and label not in include_labels:
            continue
        loaded: list[np.ndarray] = []
        for filename in filenames:
            if exclude_filenames is not None and filename in exclude_filenames:
                continue
            path = base / filename
            image = read_grayscale_image(path)
            if image is None:
                missing.append(str(path))
                continue
            _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
            loaded.append(normalize_symbol_image(binary, size=size))
        if loaded:
            templates[label] = loaded

    if missing:
        raise FileNotFoundError("missing template files: " + ", ".join(missing))
    return templates


def match_symbol(
    roi: np.ndarray,
    templates: dict[str, list[np.ndarray]],
    size: int = 64,
    unknown_threshold: float = 0.35,
) -> TemplateMatch:
    normalized = normalize_symbol_image(roi, size=size)
    candidates: dict[str, float] = {}
    for label, images in templates.items():
        scores = []
        for template in images:
            result = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)
            scores.append(float(np.max(result)))
        candidates[label] = max(scores)

    label, score = max(candidates.items(), key=lambda item: item[1])
    if score < unknown_threshold:
        return TemplateMatch(label="UNKNOWN", score=score, candidates=candidates)
    return TemplateMatch(label=label, score=score, candidates=candidates)
