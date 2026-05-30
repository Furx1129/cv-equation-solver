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
from src.vision.template_matcher import extract_geometry_features, normalize_symbol_image, precompute_template_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDWRITTEN_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "handwritten_basic"
DEFAULT_HANDWRITTEN_LABEL_DIR = PROJECT_ROOT / "data" / "labels" / "handwritten_basic"


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

        debug_base = Path(debug_dir) if debug_dir is not None else None
        debug_artifacts: dict[str, Path] = {}
        if debug_base is not None:
            debug_base.mkdir(parents=True, exist_ok=True)
            debug_artifacts["normalized"] = save_debug_image(debug_base / "normalized.png", normalized.image)
            debug_artifacts["binary"] = save_debug_image(debug_base / "binary.png", preprocessed.binary)
            (debug_base / "segments").mkdir(exist_ok=True)

        tokens: list[SymbolToken] = []
        warnings: list[str] = []
        for index, segment in enumerate(segments):
            label, score, candidates, decision_type = match_handwritten_symbol(segment.image, templates)
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
                    segment.image,
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
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return extract_geometry_features(binary)


def _disambiguate(
    query_features: dict[str, float],
    candidates: dict[str, float],
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

    stable_symbols = ("(", ")", "+", ".")
    if original_best in stable_symbols and candidates[original_best] >= 0.55:
        return adjusted, reason

    if components >= 3 and 0.75 <= aspect <= 1.3:
        force("÷", "geometry:three_components")
        return adjusted, reason
    if holes >= 2:
        force("8", "geometry:two_holes")
        return adjusted, reason

    if aspect > 2.5 and horizontal > 0.2 and max(diag_pos, diag_neg) < 0.25:
        force("-", "geometry:long_horizontal")
        return adjusted, reason
    if aspect > 1.45 and horizontal > 0.35 and max(diag_pos, diag_neg) < 0.25:
        force("=", "geometry:parallel_horizontal")
        return adjusted, reason

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


def match_handwritten_symbol(
    roi: np.ndarray,
    templates: TemplateLibrary,
    size: int = 64,
) -> tuple[str, float, dict[str, float], str]:
    """Match a handwritten symbol ROI against a template library.

    Returns (label, score, top5_candidates, decision_type) where
    decision_type is one of "template_only", "geometry_adjusted", or
    "geometry_filtered".
    """
    normalized = normalize_symbol_image(roi, size=size)
    query_features = _extract_features_at_native_resolution(roi)
    query_descriptor = _symbol_descriptor(normalized)

    candidates: dict[str, float] = {}
    for label, images in templates.images.items():
        scores: list[float] = []
        for index, template in enumerate(images):
            result = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)
            pixel_score = max(0.0, float(np.max(result)))
            descriptor_score = _descriptor_similarity(query_descriptor, templates.descriptors[label][index])
            geometry_score = _geometry_similarity(query_features, templates.features[label][index])
            scores.append(0.58 * pixel_score + 0.34 * descriptor_score + 0.08 * geometry_score)
        candidates[label] = max(scores)

    original_candidates = dict(candidates)
    adjusted, decision_type = _disambiguate(query_features, candidates)

    original_best = max(original_candidates, key=lambda k: original_candidates[k])
    adjusted_best = max(adjusted, key=lambda k: adjusted[k])

    if decision_type.startswith("geometry:"):
        decision_type = "geometry_filtered"
    elif original_best != adjusted_best:
        decision_type = "geometry_filtered"
    elif any(abs(original_candidates[k] - adjusted[k]) > 0.001 for k in adjusted):
        decision_type = "geometry_adjusted"
    else:
        decision_type = "template_only"

    label, score = max(adjusted.items(), key=lambda item: item[1])
    return label, score, dict(sorted(adjusted.items(), key=lambda item: item[1], reverse=True)[:5]), decision_type
