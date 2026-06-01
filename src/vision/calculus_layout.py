from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.expression.serializer import serialize_layout
from src.expression.types import LayoutNode, RecognitionResult, SymbolToken
from src.vision.calculus_rules import apply_calculus_geometry_rules
from src.vision.layout_analysis import tokens_to_row
from src.vision.normalization import foreground_bbox, normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image, save_debug_image
from src.vision.recognizers.handwritten_rule_template import (
    DEFAULT_HANDWRITTEN_LABEL_DIR,
    DEFAULT_HANDWRITTEN_SAMPLE_DIR,
    HandwrittenRuleTemplateRecognizer,
    TemplateLibrary,
    load_handwritten_templates,
    match_handwritten_symbol,
)
from src.vision.recognizers.printed_template import PrintedTemplateRecognizer, kind_for_label
from src.vision.segmentation import Segment, segment_characters
from src.vision.structure_2d import (
    LineCandidate,
    _content_x_range_near_line,
    _crop_binary,
    _detect_full_fraction,
    _is_empty,
    _is_fraction_line,
    _line_candidates,
    _union_bbox,
)
from src.vision.template_matcher import extract_geometry_features


def recognize_calculus_layout(
    image_path: str | Path,
    debug_dir: str | Path | None = None,
    disable_fallbacks: bool = False,
) -> RecognitionResult:
    image = read_image(image_path)
    normalized = normalize_formula_image(image)
    preprocessed = preprocess_image(
        normalized.image,
        options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
    )
    binary = _crop_binary(preprocessed.binary)

    debug_artifacts: dict[str, Path] = {}
    if debug_dir is not None:
        debug_base = Path(debug_dir)
        debug_base.mkdir(parents=True, exist_ok=True)
        debug_artifacts["binary_calculus_2d"] = save_debug_image(debug_base / "binary_calculus_2d.png", binary)

    templates = load_handwritten_templates(DEFAULT_HANDWRITTEN_SAMPLE_DIR, DEFAULT_HANDWRITTEN_LABEL_DIR)
    tokens: list[SymbolToken] = []
    layout = _detect_limit_layout(binary, templates, tokens) or _detect_integral_fraction_layout(binary, templates, tokens)
    if layout is None:
        if disable_fallbacks:
            return RecognitionResult(
                tokens=[],
                expression_text="",
                debug_artifacts=debug_artifacts,
                warnings=["calculus layout fallback disabled"],
                layout=None,
                sympy_text="",
            )
        fallback = _fallback_calculus_recognition(image_path)
        return replace(fallback, debug_artifacts={**fallback.debug_artifacts, **debug_artifacts})

    expression_text = serialize_layout(layout, target="plain")
    return RecognitionResult(
        tokens=tokens,
        expression_text=expression_text,
        debug_artifacts=debug_artifacts,
        warnings=[],
        layout=layout,
        sympy_text=serialize_layout(layout, target="sympy"),
    )


def _detect_limit_layout(
    binary: np.ndarray,
    templates: TemplateLibrary,
    tokens: list[SymbolToken],
) -> LayoutNode | None:
    line = _main_fraction_line(binary)
    if line is None:
        return None
    lx, ly, lw, lh = line.bbox
    left_split = max(1, lx - 30)
    left = binary[:, :left_split]
    if _is_empty(left):
        return None

    arrow = _condition_arrow(left)
    if arrow is None:
        return None

    variable, target = _parse_limit_condition(left, arrow, templates, tokens)
    body = _fraction_node(binary, line, templates, tokens, (0, 0))
    return LayoutNode(
        node_type="limit",
        children=[body],
        metadata={"variable": variable, "target": target},
        bbox=_union_bbox([body.bbox, _bbox_from_binary(left, (0, 0))]),
    )


def _detect_integral_fraction_layout(
    binary: np.ndarray,
    templates: TemplateLibrary,
    tokens: list[SymbolToken],
) -> LayoutNode | None:
    line = _main_fraction_line(binary)
    if line is None:
        return None
    lx, _, lw, _ = line.bbox
    left = binary[:, : max(1, lx - 20)]
    if not _looks_like_integral_sign(left):
        return None

    right = binary[:, min(binary.shape[1], lx + lw + 15) :]
    right_tokens = _plain_tokens(right, templates, (min(binary.shape[1], lx + lw + 15), 0))
    if len(right_tokens) < 2:
        return None
    if not _looks_like_differential_pair(right_tokens[-2:]):
        return None

    body = _fraction_node(binary, line, templates, tokens, (0, 0))
    tokens.extend(_integral_tokens(left))
    tokens.extend(right_tokens)
    return LayoutNode(
        node_type="integral",
        children=[body],
        metadata={"variable": right_tokens[-1].text if right_tokens[-1].text in {"x", "y"} else "x"},
        bbox=_union_bbox([body.bbox, _bbox_from_binary(left, (0, 0)), _union_bbox([token.bbox for token in right_tokens])]),
    )


def _fraction_node(
    binary: np.ndarray,
    line: LineCandidate,
    templates: TemplateLibrary,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> LayoutNode:
    x, y, w, h = line.bbox
    x0, x1 = _content_x_range_near_line(binary, line)
    gap = max(3, h)
    above = binary[:y, x0:x1]
    below_y0 = min(binary.shape[0], y + h + gap)
    below = binary[below_y0:, x0:x1]
    numerator = _parse_region(above, templates, tokens, (offset[0] + x0, offset[1]), role="numerator")
    denominator = _parse_region(below, templates, tokens, (offset[0] + x0, offset[1] + below_y0), role="denominator")
    bbox = _union_bbox([numerator.bbox, denominator.bbox, (offset[0] + x, offset[1] + y, w, h)])
    return LayoutNode(node_type="fraction", children=[numerator, denominator], bbox=bbox)


def _parse_region(
    binary: np.ndarray,
    templates: TemplateLibrary,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
    role: str = "body",
) -> LayoutNode:
    if binary.size == 0 or _is_empty(binary):
        return LayoutNode.row([])
    crop_x, crop_y = _foreground_offset(binary)
    cropped = _crop_binary(binary)
    offset = (offset[0] + crop_x, offset[1] + crop_y)

    line = _detect_full_fraction(cropped)
    if line is not None:
        return _fraction_node(cropped, line, templates, tokens, offset)

    local_tokens = _plain_tokens(cropped, templates, offset)
    if role == "numerator" and len(local_tokens) == 1 and _looks_like_single_one(local_tokens[0]):
        local_tokens[0] = replace(local_tokens[0], text="1", kind="digit", confidence=max(local_tokens[0].confidence, 0.82))
    tokens.extend(local_tokens)
    return _tokens_to_layout(local_tokens)


def _plain_tokens(binary: np.ndarray, templates: TemplateLibrary, offset: tuple[int, int]) -> list[SymbolToken]:
    if binary.size == 0 or _is_empty(binary):
        return []
    crop_x, crop_y = _foreground_offset(binary)
    cropped = _crop_binary(binary)
    try:
        segments = segment_characters(cropped, min_width=1, min_height=1, min_gap=1, pad=2)
    except ValueError:
        return []
    tokens: list[SymbolToken] = []
    for segment in _expand_segments(segments):
        tokens.append(_match_segment(segment, templates, (offset[0] + crop_x, offset[1] + crop_y)))
    return sorted(tokens, key=lambda token: (token.bbox[0], token.bbox[1]))


def _match_segment(segment: Segment, templates: TemplateLibrary, offset: tuple[int, int]) -> SymbolToken:
    label, score, candidates, _ = match_handwritten_symbol(segment.image, templates, context="calculus")
    label, score = _calculus_symbol_override(segment.image, label, score)
    x, y, w, h = segment.bbox
    return SymbolToken(
        text=label,
        kind=kind_for_label(label),
        bbox=(offset[0] + x, offset[1] + y, w, h),
        confidence=score,
        source="calculus_2d_handwritten",
        candidates=candidates,
    )


def _tokens_to_layout(tokens: list[SymbolToken]) -> LayoutNode:
    if not tokens:
        return LayoutNode.row([])
    function = _detect_function_call(tokens)
    if function is not None:
        return function
    return _detect_calculus_superscripts(tokens)


def _detect_function_call(tokens: list[SymbolToken]) -> LayoutNode | None:
    ordered = sorted(tokens, key=lambda token: token.bbox[0])
    for index, token in enumerate(ordered):
        if token.text != "(":
            continue
        close_index = _find_matching_close(ordered, index)
        if close_index is None:
            continue
        name = _classify_function_prefix("".join(t.text for t in ordered[:index]))
        if name is None:
            continue
        argument = _tokens_to_layout(ordered[index + 1 : close_index])
        return LayoutNode(
            node_type="function_call",
            children=[argument],
            metadata={"name": name},
            bbox=_union_bbox([token.bbox for token in ordered[: close_index + 1]]),
        )
    return None


def _detect_calculus_superscripts(tokens: list[SymbolToken]) -> LayoutNode:
    ordered = sorted(tokens, key=lambda token: token.bbox[0])
    children: list[LayoutNode] = []
    i = 0
    while i < len(ordered):
        base = ordered[i]
        if i + 1 < len(ordered):
            candidate = ordered[i + 1]
            if _is_superscript_candidate(base, candidate):
                children.append(
                    LayoutNode(
                        node_type="superscript",
                        children=[LayoutNode.symbol(base.text, base.bbox), LayoutNode.symbol(candidate.text, candidate.bbox)],
                        bbox=_union_bbox([base.bbox, candidate.bbox]),
                    )
                )
                i += 2
                continue
        children.append(LayoutNode.symbol(base.text, base.bbox))
        i += 1
    return LayoutNode.row(children=children, bbox=_union_bbox([token.bbox for token in ordered]))


def _is_superscript_candidate(base: SymbolToken, candidate: SymbolToken) -> bool:
    if base.text in {"+", "-", "=", "÷", "/", "(", ")"}:
        return False
    if candidate.text in {"+", "-", "=", "÷", "/", "(", ")", "."}:
        return False
    bx, by, bw, bh = base.bbox
    cx, cy, cw, ch = candidate.bbox
    if cx < bx + bw * 0.45:
        return False
    if cx > bx + bw + max(bw, bh) * 0.95:
        return False
    if ch > bh * 0.75:
        return False
    base_center = by + bh / 2
    candidate_bottom = cy + ch
    return candidate_bottom < base_center + bh * 0.08


def _parse_limit_condition(
    left: np.ndarray,
    arrow: LineCandidate,
    templates: TemplateLibrary,
    tokens: list[SymbolToken],
) -> tuple[str, str]:
    bottom_y0 = int(left.shape[0] * 0.48)
    bottom = left[bottom_y0:, :]
    ax, ay, aw, ah = arrow.bbox
    variable_roi = bottom[:, : max(1, ax - 2)]
    target_roi = bottom[:, min(bottom.shape[1], ax + aw + 2) :]
    variable = _parse_condition_variable(variable_roi, templates)
    target = _parse_condition_target(target_roi, templates)
    tokens.extend(
        [
            SymbolToken("lim", "symbol", _bbox_from_binary(left[:bottom_y0, :], (0, 0)) or (0, 0, 0, 0), 0.86, "calculus_2d_layout"),
            SymbolToken(variable, "symbol", _bbox_from_binary(variable_roi, (0, bottom_y0)) or (0, bottom_y0, 0, 0), 0.82, "calculus_2d_layout"),
            SymbolToken("->", "operator", (ax, bottom_y0 + ay, aw, ah), 0.88, "calculus_2d_layout"),
            SymbolToken(target, "symbol", _bbox_from_binary(target_roi, (ax + aw + 2, bottom_y0)) or (ax + aw + 2, bottom_y0, 0, 0), 0.82, "calculus_2d_layout"),
        ]
    )
    return variable, target


def _parse_condition_variable(binary: np.ndarray, templates: TemplateLibrary) -> str:
    tokens = _plain_tokens(binary, templates, (0, 0))
    for token in tokens:
        if token.text in {"x", "y"}:
            return token.text
    return "x"


def _parse_condition_target(binary: np.ndarray, templates: TemplateLibrary) -> str:
    if binary.size == 0 or _is_empty(binary):
        return "0"
    features = extract_geometry_features(binary)
    if features["num_holes"] >= 2:
        return "oo"
    if features["num_holes"] == 1:
        return "0"
    tokens = _plain_tokens(binary, templates, (0, 0))
    text = "".join(token.text for token in tokens)
    if any(token.text in {"8", "÷", "x"} for token in tokens) and features["aspect_ratio"] > 0.7:
        return "oo"
    if "1" in text or any(token.text == ")" and token.bbox[3] > token.bbox[2] * 1.4 for token in tokens):
        return "1"
    if "0" in text or "6" in text:
        return "0"
    return "0"


def _condition_arrow(left: np.ndarray) -> LineCandidate | None:
    bottom_y0 = int(left.shape[0] * 0.48)
    bottom = left[bottom_y0:, :]
    candidates = [
        candidate
        for candidate in _line_candidates(bottom)
        if candidate.bbox[2] >= max(35, int(left.shape[1] * 0.22)) and candidate.bbox[3] <= 12
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.bbox[2])


def _main_fraction_line(binary: np.ndarray) -> LineCandidate | None:
    lines = [line for line in _line_candidates(binary) if _is_fraction_line(binary, line)]
    if not lines:
        return None
    return max(lines, key=lambda line: line.bbox[2])


def _calculus_symbol_override(image: np.ndarray, label: str, score: float) -> tuple[str, float]:
    features = extract_geometry_features(image)
    aspect = features["aspect_ratio"]
    vertical = features["vertical_score"]
    horizontal = features["horizontal_score"]
    holes = features["num_holes"]
    components = features["num_components"]

    if label in {"y", "/"} and aspect < 0.65 and vertical > 0.15:
        return "1", max(score, 0.82)
    if label == "÷" and components <= 1 and aspect > 2.0:
        return "-", max(score, 0.84)
    if label in {"y", "4", "/"} and aspect < 0.8 and horizontal < 0.1:
        return "d", max(score, 0.78)
    if holes == 1 and 0.8 < aspect < 1.2 and label in {"8", "6", "9", "4"}:
        return "0", max(score, 0.76)
    return label, score


def _looks_like_single_one(token: SymbolToken) -> bool:
    _, _, w, h = token.bbox
    return token.text in {"1", "y", "/"} and h > w * 1.25


def _looks_like_integral_sign(binary: np.ndarray) -> bool:
    if binary.size == 0 or _is_empty(binary):
        return False
    x, y, w, h = foreground_bbox(binary)
    height, width = binary.shape
    return h >= height * 0.78 and w <= max(130, height * 0.7) and width <= max(160, height * 0.75)


def _integral_tokens(binary: np.ndarray) -> list[SymbolToken]:
    bbox = _bbox_from_binary(binary, (0, 0))
    if bbox is None:
        return []
    return [SymbolToken("∫", "symbol", bbox, 0.86, "calculus_2d_layout")]


def _looks_like_differential_pair(tokens: list[SymbolToken]) -> bool:
    if len(tokens) < 2:
        return False
    d_token, variable = tokens[-2], tokens[-1]
    _, _, dw, dh = d_token.bbox
    aspect = dw / max(1, dh)
    return variable.text in {"x", "y"} and (d_token.text == "d" or (d_token.text in {"y", "4"} and aspect < 1.1))


def _expand_segments(segments: list[Segment]) -> list[Segment]:
    expanded: list[Segment] = []
    for segment in segments:
        expanded.extend(_split_tall_disconnected_segment(segment))
    return expanded


def _split_tall_disconnected_segment(segment: Segment) -> list[Segment]:
    x, y, w, h = segment.bbox
    if h <= w * 1.8:
        return [segment]
    foreground = (segment.image < 255).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    pieces: list[Segment] = []
    for label in range(1, count):
        cx, cy, cw, ch, area = [int(value) for value in stats[label]]
        if area < 6 or cw < 2 or ch < 2:
            continue
        pad = 2
        x0 = max(0, cx - pad)
        y0 = max(0, cy - pad)
        x1 = min(segment.image.shape[1], cx + cw + pad)
        y1 = min(segment.image.shape[0], cy + ch + pad)
        pieces.append(Segment((x + x0, y + y0, x1 - x0, y1 - y0), segment.image[y0:y1, x0:x1]))
    if len(pieces) <= 1:
        return [segment]
    return sorted(pieces, key=lambda item: (item.bbox[0], item.bbox[1]))


def _classify_function_prefix(prefix: str) -> str | None:
    from src.vision.calculus_rules import _classify_function_prefix as _cfp
    return _cfp(prefix)


def _find_matching_close(tokens: list[SymbolToken], open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(tokens)):
        if tokens[index].text == "(":
            depth += 1
        elif tokens[index].text == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _foreground_offset(binary: np.ndarray) -> tuple[int, int]:
    x, y, _, _ = foreground_bbox(binary)
    return x, y


def _bbox_from_binary(binary: np.ndarray, offset: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if binary.size == 0 or _is_empty(binary):
        return None
    x, y, w, h = foreground_bbox(binary)
    return offset[0] + x, offset[1] + y, w, h


def _fallback_calculus_recognition(image_path: str | Path) -> RecognitionResult:
    printed = PrintedTemplateRecognizer().recognize(image_path=image_path)
    handwritten = HandwrittenRuleTemplateRecognizer().recognize(image_path=image_path)
    result = min((printed, handwritten), key=_recognition_penalty)
    return apply_calculus_geometry_rules(result)


def _recognition_penalty(result: RecognitionResult) -> tuple[int, int, float]:
    unknown_count = sum(1 for token in result.tokens if token.text == "UNKNOWN")
    low_confidence = sum(1 for token in result.tokens if token.confidence < 0.55)
    average_confidence = sum(token.confidence for token in result.tokens) / max(1, len(result.tokens))
    return (unknown_count, low_confidence, -average_confidence)
