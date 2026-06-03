from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.expression.normalizer import normalize_tokens
from src.expression.serializer import serialize_layout
from src.expression.types import RecognitionResult, SymbolToken
from src.vision.layout_analysis import analyze_layout
from src.vision.normalization import normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image, save_debug_image
from src.vision.recognizers.printed_template import kind_for_label, safe_label
from src.vision.segmentation import segment_characters
from src.vision.template_matcher import (
    _trim_foreground,
    extract_geometry_features,
    normalize_symbol_image,
    precompute_template_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDWRITTEN_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "handwritten_basic"
DEFAULT_HANDWRITTEN_LABEL_DIR = PROJECT_ROOT / "data" / "labels" / "handwritten_basic"

# Typical foreground pixel count on trimmed templates (rough median).
TARGET_STROKE_PIXELS = 360
LOW_CONFIDENCE_RETRY_THRESHOLD = 0.45
MATCH_WEIGHTS = (0.52, 0.36, 0.12)  # pixel, descriptor, geometry


@dataclass
class TemplateLibrary:
    images: dict[str, list[np.ndarray]]
    features: dict[str, list[dict[str, float]]]
    descriptors: dict[str, list[dict[str, np.ndarray]]]


@dataclass(frozen=True)
class HandwrittenRuleTemplateRecognizer:
    sample_dir: Path = DEFAULT_HANDWRITTEN_SAMPLE_DIR
    label_dir: Path = DEFAULT_HANDWRITTEN_LABEL_DIR
    confidence_warning_threshold: float = 0.45

    def recognize(self, image_path: str | Path, debug_dir: str | Path | None = None) -> RecognitionResult:
        templates = load_handwritten_templates(self.sample_dir, self.label_dir)
        image = read_image(image_path)
        normalized = normalize_formula_image(image)
        preprocessed = preprocess_image(
            normalized.image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
        )
        segments = segment_characters(preprocessed.binary)
        prepared_rois = prepare_equation_segments(segments)

        debug_base = Path(debug_dir) if debug_dir is not None else None
        debug_artifacts: dict[str, Path] = {}
        if debug_base is not None:
            debug_base.mkdir(parents=True, exist_ok=True)
            debug_artifacts["normalized"] = save_debug_image(debug_base / "normalized.png", normalized.image)
            debug_artifacts["binary"] = save_debug_image(debug_base / "binary.png", preprocessed.binary)
            (debug_base / "segments").mkdir(exist_ok=True)

        tokens: list[SymbolToken] = []
        warnings: list[str] = []
        for index, (segment, prepared_roi) in enumerate(zip(segments, prepared_rois)):
            label, score, candidates, decision_type = match_handwritten_symbol(prepared_roi, templates)
            token = SymbolToken(
                text=label,
                kind=kind_for_label(label),
                bbox=segment.bbox,
                confidence=score,
                source="handwritten_rule_template",
                candidates=candidates,
            )
            tokens.append(token)
            if score < self.confidence_warning_threshold:
                warnings.append(f"low confidence handwritten token {index}: {label} score={score:.3f}")
            if decision_type != "template_only":
                warnings.append(f"handwritten token {index}: decision={decision_type}")
            if debug_base is not None:
                save_debug_image(
                    debug_base / "segments" / f"segment_{index:02d}_{safe_label(label)}.png",
                    prepared_roi,
                )

        expression = normalize_tokens(tokens)
        display_text = "".join(token.text for token in tokens)
        layout = analyze_layout(tokens)
        if debug_base is not None:
            table_path = debug_base / "tokens.csv"
            with table_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "text", "kind", "bbox", "confidence", "source"])
                for index, token in enumerate(tokens):
                    writer.writerow([index, token.text, token.kind, token.bbox, f"{token.confidence:.6f}", token.source])
            debug_artifacts["tokens"] = table_path

        return RecognitionResult(
            tokens=tokens,
            expression_text=display_text,
            debug_artifacts=debug_artifacts,
            warnings=warnings + expression.warnings,
            layout=layout,
            sympy_text=serialize_layout(layout, target="sympy"),
        )


def _foreground_fill_ratio(binary: np.ndarray) -> float:
    mask = binary < 128
    return float(mask.sum()) / max(1, mask.size)


def refine_segment_binary(roi: np.ndarray) -> np.ndarray:
    """Per-segment binarization aligned with template loading."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    if roi.ndim == 2 and float(np.mean(roi > 200)) > 0.60:
        binary = roi.copy()
        if float(np.mean(binary < 128)) > 0.5:
            binary = 255 - binary
        trimmed = _trim_foreground(binary)
        if trimmed.size == 0:
            return binary
        closed = cv2.morphologyEx(trimmed, cv2.MORPH_CLOSE, kernel, iterations=1)
        return _pad_binary_trimmed(closed, pad=2)

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if float(np.mean(gray < 128)) > 0.5:
        gray = 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _pad_binary_trimmed(_trim_foreground(closed), pad=2)


def unify_stroke_fill_ratio(
    binary: np.ndarray,
    target_pixels: int = TARGET_STROKE_PIXELS,
) -> np.ndarray:
    """Adjust stroke thickness so pixel count is closer to single-char templates."""
    trimmed = _trim_foreground(binary)
    if trimmed.size == 0 or trimmed.shape[0] == 0 or trimmed.shape[1] == 0:
        return binary

    area = int((trimmed < 128).sum())
    if int(target_pixels * 0.75) <= area <= int(target_pixels * 1.35):
        return _pad_binary_trimmed(trimmed, pad=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    if area < int(target_pixels * 0.75):
        if area < int(target_pixels * 0.35):
            scale = min(2.0, float(np.sqrt(target_pixels / max(1, area))))
            new_h = max(1, int(round(trimmed.shape[0] * scale)))
            new_w = max(1, int(round(trimmed.shape[1] * scale)))
            trimmed = cv2.resize(trimmed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            iterations = 2 if area < int(target_pixels * 0.5) else 1
            trimmed = cv2.dilate(trimmed, kernel, iterations=iterations)
    else:
        trimmed = cv2.erode(trimmed, kernel, iterations=1)
    return _pad_binary_trimmed(trimmed, pad=2)


def _pad_binary_trimmed(trimmed: np.ndarray, pad: int = 2) -> np.ndarray:
    h, w = trimmed.shape[:2]
    canvas = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = trimmed
    return canvas


def align_segment_heights(binaries: list[np.ndarray], pad: int = 2) -> list[np.ndarray]:
    """Second trim + uniform stroke height across all segments in one expression."""
    trimmed = [_trim_foreground(binary) for binary in binaries]
    heights = [image.shape[0] for image in trimmed if image.shape[0] > 0]
    if not heights:
        return binaries

    target_height = max(4, int(np.median(heights)))
    aligned: list[np.ndarray] = []
    for image in trimmed:
        if image.shape[0] == 0 or image.shape[1] == 0:
            aligned.append(image)
            continue
        scale = target_height / image.shape[0]
        new_w = max(1, int(round(image.shape[1] * scale)))
        resized = cv2.resize(
            image,
            (new_w, target_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        aligned.append(_pad_binary_trimmed(resized, pad=pad))
    return aligned


def prepare_equation_segments(segments) -> list[np.ndarray]:
    """Local template-style binarization, stroke fill, and batch height alignment."""
    refined = [refine_segment_binary(segment.image) for segment in segments]
    unified = [unify_stroke_fill_ratio(binary) for binary in refined]
    return align_segment_heights(unified)


def _segment_match_variants(binary: np.ndarray) -> list[np.ndarray]:
    variants = [binary]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    if not np.array_equal(dilated, binary):
        variants.append(dilated)
    scaled = _scale_binary(binary, 1.2)
    if scaled is not None:
        variants.append(scaled)
    return variants


def _scale_binary(binary: np.ndarray, factor: float) -> np.ndarray | None:
    trimmed = _trim_foreground(binary)
    if trimmed.shape[0] == 0 or trimmed.shape[1] == 0:
        return None
    new_h = max(1, int(round(trimmed.shape[0] * factor)))
    new_w = max(1, int(round(trimmed.shape[1] * factor)))
    resized = cv2.resize(trimmed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _pad_binary_trimmed(resized, pad=2)


def load_handwritten_templates(
    sample_dir: str | Path,
    label_dir: str | Path,
    size: int = 64,
    exclude_stem: str | None = None,
) -> TemplateLibrary:
    samples = Path(sample_dir)
    labels = Path(label_dir)
    images: dict[str, list[np.ndarray]] = {}
    for image_path in sorted(samples.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if exclude_stem is not None and image_path.stem == exclude_stem:
            continue
        label_path = labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        label = label_path.read_text(encoding="utf-8").strip()
        data = np.fromfile(str(image_path), dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images.setdefault(label, []).append(normalize_symbol_image(binary, size=size))
    if not images:
        raise FileNotFoundError(f"no handwritten templates found in {samples}")
    return TemplateLibrary(
        images=images,
        features=precompute_template_features(images),
        descriptors=_precompute_template_descriptors(images),
    )


def _extract_features_at_native_resolution(roi: np.ndarray) -> dict[str, float]:
    if roi.ndim == 3:
        binary = refine_segment_binary(roi)
    elif roi.ndim == 2 and roi.max() <= 1:
        binary = (roi * 255).astype(np.uint8)
    else:
        binary = refine_segment_binary(roi) if float(np.mean(roi < 128)) < 0.5 else roi
    return extract_geometry_features(binary)


def _disambiguate(
    query_features: dict[str, float],
    candidates: dict[str, float],
    context: str = "handwritten",
) -> tuple[dict[str, float], str]:
    """Adjust template scores using deterministic handwriting geometry."""
    adjusted = dict(candidates)
    reason = "template_only"
    original_best = max(candidates, key=lambda key: candidates[key])

    def penalize(labels: tuple[str, ...], factor: float) -> None:
        for label in labels:
            if label in adjusted:
                adjusted[label] *= factor

    def boost(label: str, amount: float) -> None:
        if label in adjusted:
            adjusted[label] += amount

    def force(label: str, rule: str) -> None:
        nonlocal reason
        if label not in adjusted:
            return
        adjusted[label] = max(adjusted.values()) + 0.08
        reason = rule

    holes = int(round(query_features["num_holes"]))
    components = int(round(query_features["num_components"]))
    aspect = query_features["aspect_ratio"]
    centroid_x = query_features["centroid_x"]
    centroid_y = query_features["centroid_y"]
    top_bottom = query_features["top_bottom_ratio"]
    left_right = query_features["left_right_ratio"]
    horizontal = query_features["horizontal_score"]
    vertical = query_features["vertical_score"]
    diag_pos = query_features["diag_pos_score"]
    diag_neg = query_features["diag_neg_score"]
    diagonal_balance = min(diag_pos, diag_neg) / max(max(diag_pos, diag_neg), 1e-6)

    if components >= 3 and 0.75 <= aspect <= 1.3:
        force("÷", "geometry:three_components")
        return adjusted, reason
    if holes >= 2:
        force("8", "geometry:two_holes")
        return adjusted, reason

    stable_symbols = ("(", ")", "+", ".")
    if original_best in stable_symbols and candidates[original_best] >= 0.55:
        return adjusted, reason

    if aspect > 2.5 and horizontal > 0.2 and max(diag_pos, diag_neg) < 0.25:
        force("-", "geometry:long_horizontal")
        return adjusted, reason
    if aspect > 1.45 and horizontal > 0.35 and max(diag_pos, diag_neg) < 0.25:
        force("=", "geometry:parallel_horizontal")
        return adjusted, reason

    # ---- Layer B: context-aware adjustments ----
    if context == "calculus":
        boost("d", 0.10)
        boost("x", 0.05)
        penalize(("4", "6"), 0.75)

    digit_score = max((candidates.get(label, 0.0) for label in "0123456789"), default=0.0)
    nondigit_score = max(
        (candidates.get(label, 0.0) for label in ("+", "-", "÷", "/", "=", "(", ")", ".", "x", "y", "×")),
        default=0.0,
    )
    digit_context = digit_score >= nondigit_score - 0.04

    if holes == 1:
        penalize(("2", "3", "5", "8"), 0.35)
        if top_bottom > 1.08 and left_right < 0.95 and centroid_y < 0.5 and digit_context:
            force("9", "geometry:upper_right_loop")
        elif left_right > 1.1 and centroid_y >= 0.49 and digit_context:
            force("6", "geometry:lower_left_loop")
        elif 0.85 <= top_bottom <= 1.15 and 0.85 <= left_right <= 1.25 and digit_context:
            if 0.60 < aspect < 0.85:
                force("0", "geometry:tall_zero")
            elif 0.85 <= aspect <= 1.05:
                force("o", "geometry:round_oh")
            elif 1.0 < aspect <= 1.25:
                force("O", "geometry:wide_oh_capital")
            else:
                force("0", "geometry:centered_loop")
        elif left_right < 0.75 and digit_context:
            force("4", "geometry:right_heavy_open_four")
        return adjusted, reason

    if digit_context:
        penalize(("0", "6", "8", "9"), 0.45)

    if (
        components < 3
        and 0.8 <= aspect <= 0.9
        and 0.9 <= top_bottom <= 1.2
        and 0.78 <= left_right <= 0.98
        and query_features["fill_ratio"] > 0.28
        and diag_neg > 0.7
        and candidates.get("4", 0.0) >= candidates.get("÷", 0.0) - 0.05
    ):
        force("4", "geometry:open_four")
        return adjusted, reason

    if aspect < 0.72 and top_bottom > 1.45 and diag_pos > 0.2 and diag_neg > 0.3 and candidates.get("y", 0.0) > 0.25:
        force("y", "geometry:y_upper_fork")
        return adjusted, reason
    if (
        aspect < 0.68
        and centroid_x < 0.65
        and vertical > 0.45
        and diag_neg > 0.18
        and horizontal < 0.25
        and candidates.get("y", 0.0) > 0.25
    ):
        force("y", "geometry:y_descender")
        return adjusted, reason
    if (
        aspect > 0.75
        and top_bottom > 1.25
        and query_features["fill_ratio"] < 0.25
        and candidates.get("y", 0.0) >= max(candidates.get("x", 0.0), candidates.get("7", 0.0)) - 0.06
    ):
        force("y", "geometry:y_open_tail")
        return adjusted, reason

    if diag_neg > 1.35 and diag_pos < 0.25 and horizontal < 0.12 and vertical < 0.12 and 0.8 <= top_bottom <= 1.2:
        force("/", "geometry:pure_slash")
        return adjusted, reason
    if aspect < 0.55 and diag_neg > 0.4 and vertical > 0.12 and horizontal < 0.25 and top_bottom > 1.2:
        force("1", "geometry:narrow_one")
        return adjusted, reason
    if diag_pos > 0.2 and diag_neg > 0.2 and horizontal < 0.15 and max(candidates.get("x", 0.0), candidates.get("×", 0.0)) > 0.25:
        if 0.9 <= aspect <= 1.08 and diagonal_balance > 0.5 and abs(centroid_x - 0.5) < 0.06:
            if candidates.get("×", 0.0) - candidates.get("x", 0.0) < 0.08:
                force("x", "geometry:near_tie_variable_x")
            else:
                force("×", "geometry:centered_cross")
        else:
            force("x", "geometry:variable_x")
        return adjusted, reason
    if aspect > 1.1 and diag_neg > 0.4 and horizontal < 0.15 and vertical < 0.15:
        force("x", "geometry:wide_variable_x")
        return adjusted, reason

    if digit_context and horizontal > 0.3 and diag_neg > 0.6 and left_right >= 0.9 and top_bottom < 1.0:
        force("2", "geometry:two_bottom_sweep")
        return adjusted, reason
    if (
        digit_context
        and left_right < 0.75
        and top_bottom > 0.88
        and vertical < 0.1
        and candidates.get("3", 0.0) >= candidates.get("5", 0.0) - 0.08
    ):
        force("3", "geometry:right_heavy_three")
        return adjusted, reason

    if query_features["density_top"] > 0.2:
        boost("7", 0.04)
        boost("5", 0.04)
    if query_features["density_bottom"] > 0.2:
        boost("y", 0.04)

    # ---- new confusion pair rules ----
    # , vs .
    if candidates.get(",", 0.0) > 0.30 and candidates.get(".", 0.0) > 0.30:
        if centroid_y > 0.60 and query_features["fill_ratio"] < 0.10:
            force(",", "geometry:low_comma")
        elif query_features["fill_ratio"] < 0.08 and centroid_y < 0.55:
            force(".", "geometry:small_dot")

    # 2 vs z
    if candidates.get("2", 0.0) > 0.30 and candidates.get("z", 0.0) > 0.30:
        if top_bottom < 0.85:
            force("2", "geometry:curved_top_two")
        elif top_bottom > 0.95:
            force("z", "geometry:flat_top_z")

    # 5 vs s
    if candidates.get("5", 0.0) > 0.30 and candidates.get("s", 0.0) > 0.30:
        if query_features["density_top"] > 0.15:
            force("5", "geometry:top_bar_five")
        else:
            force("s", "geometry:no_top_bar_s")

    # 9 vs g
    if candidates.get("9", 0.0) > 0.30 and candidates.get("g", 0.0) > 0.30:
        if query_features["density_bottom"] < 0.40:
            force("9", "geometry:closed_bottom_nine")
        elif query_features["density_bottom"] > 0.50:
            force("g", "geometry:descender_g")

    # 6 vs b
    if candidates.get("6", 0.0) > 0.30 and candidates.get("b", 0.0) > 0.30:
        if left_right > 1.05:
            force("6", "geometry:left_loop_six")
        elif left_right < 0.95:
            force("b", "geometry:right_loop_b")

    # - vs _
    if candidates.get("-", 0.0) > 0.30 and candidates.get("_", 0.0) > 0.30:
        if centroid_y > 0.65:
            force("_", "geometry:low_underscore")
        elif 0.40 < centroid_y < 0.58:
            force("-", "geometry:mid_minus")

    if reason == "template_only" and any(abs(candidates[k] - adjusted[k]) > 0.001 for k in adjusted):
        reason = "geometry_adjusted"
    return adjusted, reason


def _precompute_template_descriptors(
    templates: dict[str, list[np.ndarray]],
) -> dict[str, list[dict[str, np.ndarray]]]:
    return {label: [_symbol_descriptor(image) for image in images] for label, images in templates.items()}


def _symbol_descriptor(image: np.ndarray) -> dict[str, np.ndarray]:
    normalized = normalize_symbol_image(image)
    foreground = (normalized < 128).astype(np.float32)
    zones = []
    for y in range(0, 64, 8):
        for x in range(0, 64, 8):
            zones.append(float(np.mean(foreground[y : y + 8, x : x + 8])))
    gx = cv2.Sobel(foreground, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(foreground, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = np.zeros(8, dtype=np.float32)
    bins = np.floor((angle % 180) / 22.5).astype(np.int32)
    for bin_index, weight in zip(bins.reshape(-1), mag.reshape(-1)):
        hist[min(7, int(bin_index))] += float(weight)
    if hist.sum() > 0:
        hist /= hist.sum()
    return {
        "row": foreground.mean(axis=1),
        "col": foreground.mean(axis=0),
        "zones": np.asarray(zones, dtype=np.float32),
        "gradient": hist,
    }


def _descriptor_similarity(query: dict[str, np.ndarray], template: dict[str, np.ndarray]) -> float:
    projection = (
        _cosine_similarity(query["row"], template["row"]) + _cosine_similarity(query["col"], template["col"])
    ) / 2.0
    zone = 1.0 - float(np.mean(np.abs(query["zones"] - template["zones"])))
    gradient = _cosine_similarity(query["gradient"], template["gradient"])
    return max(0.0, min(1.0, 0.35 * projection + 0.45 * zone + 0.20 * gradient))


def _geometry_similarity(query: dict[str, float], template: dict[str, float]) -> float:
    keys = (
        "num_holes",
        "aspect_ratio",
        "centroid_x",
        "centroid_y",
        "top_bottom_ratio",
        "left_right_ratio",
        "fill_ratio",
    )
    weights = np.asarray([0.3, 0.7, 0.2, 0.2, 0.25, 0.25, 0.2], dtype=np.float32)
    delta = np.asarray([abs(query[key] - template[key]) for key in keys], dtype=np.float32)
    return float(np.exp(-float(np.dot(delta, weights))))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 1.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denom)))


def _score_variant_against_templates(
    binary: np.ndarray,
    templates: TemplateLibrary,
    size: int,
    pixel_w: float,
    desc_w: float,
    geom_w: float,
) -> dict[str, float]:
    normalized = normalize_symbol_image(binary, size=size)
    query_features = extract_geometry_features(binary)
    query_descriptor = _symbol_descriptor(normalized)

    candidates: dict[str, float] = {}
    for label, images in templates.images.items():
        scores: list[float] = []
        for index, template in enumerate(images):
            result = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)
            pixel_score = max(0.0, float(np.max(result)))
            descriptor_score = _descriptor_similarity(query_descriptor, templates.descriptors[label][index])
            geometry_score = _geometry_similarity(query_features, templates.features[label][index])
            scores.append(pixel_w * pixel_score + desc_w * descriptor_score + geom_w * geometry_score)
        candidates[label] = max(scores)
    return candidates


def _is_prepared_segment_binary(roi: np.ndarray) -> bool:
    return roi.ndim == 2 and float(np.mean(roi > 200)) > 0.70


def match_handwritten_symbol(
    roi: np.ndarray,
    templates: TemplateLibrary,
    size: int = 64,
    context: str = "handwritten",
) -> tuple[str, float, dict[str, float], str]:
    """Match a handwritten symbol ROI against a template library.

    Returns (label, score, top5_candidates, decision_type) where
    decision_type is one of "template_only", "geometry_adjusted", or
    "geometry_filtered".
    """
    if _is_prepared_segment_binary(roi):
        binary = roi.copy()
    else:
        binary = refine_segment_binary(roi)

    pixel_w, desc_w, geom_w = MATCH_WEIGHTS
    merged: dict[str, float] = {}
    query_features = extract_geometry_features(binary)

    primary = _score_variant_against_templates(binary, templates, size, pixel_w, desc_w, geom_w)
    merged = dict(primary)
    primary_best = max(primary.values()) if primary else 0.0

    if primary_best < 0.58:
        for variant in _segment_match_variants(binary)[1:]:
            for label, score in _score_variant_against_templates(
                variant, templates, size, pixel_w, desc_w, geom_w
            ).items():
                merged[label] = max(merged.get(label, 0.0), score)

    candidates = merged
    original_candidates = dict(candidates)
    adjusted, decision_type = _disambiguate(query_features, candidates, context=context)

    label, score = max(adjusted.items(), key=lambda item: item[1])
    if score < LOW_CONFIDENCE_RETRY_THRESHOLD:
        retry_binary = _scale_binary(binary, 1.25)
        if retry_binary is not None:
            retry_merged: dict[str, float] = {}
            for variant in _segment_match_variants(retry_binary):
                for retry_label, retry_score in _score_variant_against_templates(
                    variant, templates, size, pixel_w, desc_w, geom_w
                ).items():
                    retry_merged[retry_label] = max(retry_merged.get(retry_label, 0.0), retry_score)
            retry_adjusted, retry_reason = _disambiguate(query_features, retry_merged, context=context)
            retry_label, retry_score = max(retry_adjusted.items(), key=lambda item: item[1])
            if retry_score > score:
                label, score = retry_label, retry_score
                adjusted = retry_adjusted
                original_candidates = retry_merged
                decision_type = retry_reason

    original_best = max(original_candidates, key=lambda k: original_candidates[k])
    if decision_type.startswith("geometry:"):
        decision_type = "geometry_filtered"
    elif original_best != label:
        decision_type = "geometry_filtered"
    elif any(abs(original_candidates[k] - adjusted[k]) > 0.001 for k in adjusted):
        decision_type = "geometry_adjusted"
    else:
        decision_type = "template_only"

    return label, score, dict(sorted(adjusted.items(), key=lambda item: item[1], reverse=True)[:5]), decision_type
