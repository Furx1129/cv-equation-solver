from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.expression.serializer import serialize_layout
from src.expression.types import LayoutNode, RecognitionResult, SymbolToken
from src.vision.normalization import foreground_bbox, normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image, save_debug_image
from src.vision.segmentation import Segment, segment_characters
from src.vision.template_matcher import load_templates, match_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates" / "printed_basic"
STRUCTURE_TEMPLATE_LABELS = frozenset("0123456789+-x()=.ay")


@dataclass(frozen=True)
class StructureAtom:
    bbox: tuple[int, int, int, int]
    node: LayoutNode


@dataclass(frozen=True)
class LineCandidate:
    bbox: tuple[int, int, int, int]
    source: str = "component"


@dataclass(frozen=True)
class Structure2DEngine:
    token_source: str
    match_segments: Callable[[list[Segment], tuple[int, int]], list[SymbolToken]]
    split_tall_segments: bool = True


def create_printed_engine(templates: dict[str, list[np.ndarray]]) -> Structure2DEngine:
    def match_segments(segments: list[Segment], offset: tuple[int, int]) -> list[SymbolToken]:
        tokens: list[SymbolToken] = []
        for segment in segments:
            match = match_symbol(segment.image, templates)
            x, y, w, h = segment.bbox
            label = _geometry_override(segment.image, match.label, match.score)
            score = 0.86 if label != match.label else match.score
            tokens.append(
                SymbolToken(
                    text=label,
                    kind=_kind_for_label(label),
                    bbox=(offset[0] + x, offset[1] + y, w, h),
                    confidence=score,
                    source="structure_2d_template",
                    candidates=dict(sorted(match.candidates.items())),
                )
            )
        return tokens

    return Structure2DEngine(token_source="structure_2d_template", match_segments=match_segments)


def create_handwritten_engine(templates) -> Structure2DEngine:
    from src.vision.recognizers.handwritten_rule_template import (
        kind_for_label,
        match_handwritten_symbol,
        prepare_equation_segments,
    )

    def match_segments(segments: list[Segment], offset: tuple[int, int]) -> list[SymbolToken]:
        if not segments:
            return []
        prepared = prepare_equation_segments(segments)
        tokens: list[SymbolToken] = []
        for segment, roi in zip(segments, prepared):
            label, score, candidates, _ = match_handwritten_symbol(roi, templates, context="handwritten_2d")
            x, y, w, h = segment.bbox
            tokens.append(
                SymbolToken(
                    text=label,
                    kind=kind_for_label(label),
                    bbox=(offset[0] + x, offset[1] + y, w, h),
                    confidence=score,
                    source="handwritten_2d_structure",
                    candidates=candidates,
                )
            )
        return tokens

    return Structure2DEngine(token_source="handwritten_2d_structure", match_segments=match_segments, split_tall_segments=False)


def looks_like_handwritten_2d_structure(binary: np.ndarray) -> bool:
    """Handwritten gate: only enter structure_2d for clear full fractions.

    Single symbols, superscripts, and subscripts stay on the flat handwritten
    pipeline (column segmentation + analyze_layout) to avoid over-splitting strokes.
    """
    cropped = _crop_binary(binary)
    if _is_empty(cropped):
        return False
    return _detect_full_fraction(cropped) is not None


def looks_like_2d_structure(binary: np.ndarray) -> bool:
    cropped = _crop_binary(binary)
    if _is_empty(cropped):
        return False
    if _detect_full_fraction(cropped) is not None:
        return True
    boxes = _meaningful_component_boxes(cropped)
    if _has_script_layout(boxes):
        return True
    if _count_stacked_pairs(boxes) > 1:
        return True
    return False


def recognize_2d_layout(image_path: str | Path, debug_dir: str | Path | None = None) -> RecognitionResult:
    image = read_image(image_path)
    normalized = normalize_formula_image(image)
    preprocessed = preprocess_image(
        normalized.image,
        options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
    )
    binary = _crop_binary(preprocessed.binary)
    templates = load_templates(
        DEFAULT_TEMPLATE_DIR,
        include_labels=STRUCTURE_TEMPLATE_LABELS,
        exclude_filenames=frozenset({"tem_mul_x.png"}),
    )
    engine = create_printed_engine(templates)
    return recognize_2d_from_binary(binary, engine, debug_dir=debug_dir, debug_key="binary_2d")


def recognize_2d_from_binary(
    binary: np.ndarray,
    engine: Structure2DEngine,
    debug_dir: str | Path | None = None,
    debug_key: str = "binary_2d",
) -> RecognitionResult:
    debug_artifacts: dict[str, Path] = {}
    if debug_dir is not None:
        debug_base = Path(debug_dir)
        debug_base.mkdir(parents=True, exist_ok=True)
        debug_artifacts[debug_key] = save_debug_image(debug_base / f"{debug_key}.png", binary)

    tokens: list[SymbolToken] = []
    layout = _parse_region(binary, engine, tokens=tokens, offset=(0, 0))
    expression_text = serialize_layout(layout, target="plain")
    return RecognitionResult(
        tokens=tokens,
        expression_text=expression_text,
        debug_artifacts=debug_artifacts,
        warnings=[],
        layout=layout,
        sympy_text=serialize_layout(layout, target="sympy"),
    )


def _parse_region(
    binary: np.ndarray,
    engine: Structure2DEngine,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> LayoutNode:
    if _is_empty(binary):
        return LayoutNode.row([])
    crop_x, crop_y = _foreground_offset(binary)
    binary = _crop_binary(binary)
    offset = (offset[0] + crop_x, offset[1] + crop_y)

    full_fraction = _detect_full_fraction(binary)
    if full_fraction is not None:
        return _fraction_node(binary, full_fraction, engine, tokens, offset)

    atoms = _detect_structure_atoms(binary, engine, tokens, offset)
    children: list[LayoutNode] = []
    cursor = 0
    for atom in atoms:
        x, _, w, _ = atom.bbox
        if x > cursor:
            children.extend(_plain_children(binary[:, cursor:x], engine, tokens, (offset[0] + cursor, offset[1])))
        children.append(atom.node)
        cursor = max(cursor, x + w)
    if cursor < binary.shape[1]:
        children.extend(_plain_children(binary[:, cursor:], engine, tokens, (offset[0] + cursor, offset[1])))

    if not children:
        children = _plain_children(binary, engine, tokens, offset)
    return LayoutNode.row(children=children, bbox=_bbox_from_binary(binary, offset))


def _detect_structure_atoms(
    binary: np.ndarray,
    engine: Structure2DEngine,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> list[StructureAtom]:
    atoms: list[StructureAtom] = []
    occupied: list[tuple[int, int, int, int]] = []
    for line in _line_candidates(binary):
        if _overlaps_any(line.bbox, occupied):
            continue
        if _is_fraction_line(binary, line):
            node = _fraction_node(binary, line, engine, tokens, offset)
            bbox = node.bbox or line.bbox
            local_bbox = (bbox[0] - offset[0], bbox[1] - offset[1], bbox[2], bbox[3])
            atoms.append(StructureAtom(local_bbox, node))
            occupied.append(local_bbox)
            continue
        sqrt_line = _as_sqrt_line(binary, line)
        if sqrt_line is not None:
            node = _sqrt_node(binary, sqrt_line, engine, tokens, offset)
            bbox = node.bbox or sqrt_line.bbox
            local_bbox = (bbox[0] - offset[0], bbox[1] - offset[1], bbox[2], bbox[3])
            atoms.append(StructureAtom(local_bbox, node))
            occupied.append(local_bbox)
    return sorted(atoms, key=lambda atom: atom.bbox[0])


def _fraction_node(
    binary: np.ndarray,
    line: LineCandidate,
    engine: Structure2DEngine,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> LayoutNode:
    x, y, w, h = line.bbox
    x0, x1 = _content_x_range_near_line(binary, line)
    gap = max(3, h)
    above_y1 = y
    below_y0 = min(binary.shape[0], y + h + gap)
    above = binary[:above_y1, x0:x1]
    below = binary[below_y0:, x0:x1]
    numerator = _parse_region(above, engine, tokens, (offset[0] + x0, offset[1]))
    denominator = _parse_region(below, engine, tokens, (offset[0] + x0, offset[1] + below_y0))
    bbox = _union_bbox([numerator.bbox, denominator.bbox, (offset[0] + x, offset[1] + y, w, h)])
    return LayoutNode(node_type="fraction", children=[numerator, denominator], bbox=bbox)


def _sqrt_node(
    binary: np.ndarray,
    line: LineCandidate,
    engine: Structure2DEngine,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> LayoutNode:
    x, y, w, h = line.bbox
    left = _sqrt_left_edge(binary, line)
    inner_x0 = min(binary.shape[1], x + 10)
    inner_x1 = min(binary.shape[1], x + w + 4)
    inner_y0 = min(binary.shape[0], y + h + 3)
    inner = binary[inner_y0:, inner_x0:inner_x1]
    child = _parse_region(inner, engine, tokens, (offset[0] + inner_x0, offset[1] + inner_y0))
    bbox = _union_bbox([child.bbox, (offset[0] + left, offset[1], inner_x1 - left, binary.shape[0])])
    return LayoutNode(node_type="sqrt", children=[child], bbox=bbox)


def _plain_children(
    binary: np.ndarray,
    engine: Structure2DEngine,
    tokens: list[SymbolToken],
    offset: tuple[int, int],
) -> list[LayoutNode]:
    binary = _crop_binary(binary)
    if _is_empty(binary):
        return []
    local_offset = _foreground_offset(binary)
    cropped = _crop_binary(binary)
    segments = segment_characters(cropped, min_width=1, min_height=1, min_gap=1, pad=2)
    if engine.split_tall_segments:
        expanded_segments: list[Segment] = []
        for segment in segments:
            expanded_segments.extend(_split_tall_disconnected_segment(segment))
    else:
        expanded_segments = segments
    segment_offset = (offset[0] + local_offset[0], offset[1] + local_offset[1])
    local_tokens = engine.match_segments(expanded_segments, segment_offset)
    tokens.extend(local_tokens)
    return _tokens_to_script_nodes(local_tokens)


def _split_tall_disconnected_segment(segment: Segment) -> list[Segment]:
    x, y, w, h = segment.bbox
    if h <= w * 1.8:
        return [segment]
    foreground = (segment.image < 255).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    pieces: list[Segment] = []
    for label in range(1, count):
        cx, cy, cw, ch, area = [int(v) for v in stats[label]]
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


def _tokens_to_script_nodes(tokens: list[SymbolToken]) -> list[LayoutNode]:
    ordered = sorted(tokens, key=lambda t: (t.bbox[0], t.bbox[1]))
    children: list[LayoutNode] = []
    i = 0
    while i < len(ordered):
        base = ordered[i]
        node = LayoutNode.symbol(base.text, base.bbox)
        script_indexes: list[int] = []
        sup: SymbolToken | None = None
        sub: SymbolToken | None = None
        j = i + 1
        while j < len(ordered):
            candidate = ordered[j]
            if not _is_script_candidate(base, candidate):
                break
            relation = _script_relation(base, candidate)
            if relation == "sup" and sup is None:
                sup = candidate
                script_indexes.append(j)
            elif relation == "sub" and sub is None:
                sub = candidate
                script_indexes.append(j)
            j += 1

        if sub is not None:
            node = LayoutNode(
                node_type="subscript",
                children=[node, LayoutNode.symbol(sub.text, sub.bbox)],
                bbox=_union_bbox([node.bbox, sub.bbox]),
            )
        if sup is not None:
            node = LayoutNode(
                node_type="superscript",
                children=[node, LayoutNode.symbol(sup.text, sup.bbox)],
                bbox=_union_bbox([node.bbox, sup.bbox]),
            )
        children.append(node)
        i = max(script_indexes) + 1 if script_indexes else i + 1
    return children


def _is_script_candidate(base: SymbolToken, candidate: SymbolToken) -> bool:
    if base.text in {"+", "-", "=", "(", ")"}:
        return False
    if candidate.text in {"+", "-", "=", "(", ")", "÷"}:
        return False
    bx, by, bw, bh = base.bbox
    cx, cy, cw, ch = candidate.bbox
    if cx < bx + bw * 0.45:
        return False
    if cx > bx + bw + max(bh, bw) * 1.1:
        return False
    if ch > bh * 0.95:
        return False
    return _script_relation(base, candidate) is not None


def _script_relation(base: SymbolToken, candidate: SymbolToken) -> str | None:
    _, by, _, bh = base.bbox
    _, cy, _, ch = candidate.bbox
    base_center = by + bh / 2
    cand_center = cy + ch / 2
    cand_bottom = cy + ch
    if ch < bh * 0.90 and cand_bottom > base_center:
        return "sub"
    if cand_center < base_center - bh * 0.10:
        return "sup"
    if cand_center > base_center + bh * 0.12:
        return "sub"
    if ch < bh * 0.90 and cy > by - bh * 0.10:
        return "sub"
    return None


def _line_candidates(binary: np.ndarray) -> list[LineCandidate]:
    foreground = (binary < 255).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    candidates: list[LineCandidate] = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if w >= 28 and h <= max(8, int(binary.shape[0] * 0.08)) and w / max(1, h) >= 5:
            candidates.append(LineCandidate((x, y, w, h), source="component"))
    candidates.extend(_projection_line_candidates(binary))
    return _dedupe_lines(candidates)


def _projection_line_candidates(binary: np.ndarray) -> list[LineCandidate]:
    foreground = binary < 255
    candidates: list[LineCandidate] = []
    for y in range(binary.shape[0]):
        cols = np.where(foreground[y])[0]
        if cols.size == 0:
            continue
        start = int(cols[0])
        prev = int(cols[0])
        for col in map(int, cols[1:]):
            if col - prev > 1:
                _append_projection_run(candidates, binary, start, prev, y)
                start = col
            prev = col
        _append_projection_run(candidates, binary, start, prev, y)
    return candidates


def _append_projection_run(candidates: list[LineCandidate], binary: np.ndarray, x1: int, x2: int, y: int) -> None:
    width = x2 - x1 + 1
    if width < 35:
        return
    y0 = y
    y1 = y
    foreground = binary < 255
    while y0 > 0 and foreground[y0 - 1, x1:x2 + 1].sum() > width * 0.65:
        y0 -= 1
    while y1 + 1 < binary.shape[0] and foreground[y1 + 1, x1:x2 + 1].sum() > width * 0.65:
        y1 += 1
    h = y1 - y0 + 1
    if h <= max(8, int(binary.shape[0] * 0.08)):
        candidates.append(LineCandidate((x1, y0, width, h), source="projection"))


def _dedupe_lines(candidates: list[LineCandidate]) -> list[LineCandidate]:
    result: list[LineCandidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.bbox[0], c.bbox[1], -c.bbox[2])):
        if not any(_iou(candidate.bbox, existing.bbox) > 0.55 for existing in result):
            result.append(candidate)
    return sorted(result, key=lambda c: c.bbox[2], reverse=True)


def _detect_full_fraction(binary: np.ndarray) -> LineCandidate | None:
    candidates = [line for line in _line_candidates(binary) if _is_fraction_line(binary, line)]
    if not candidates:
        return None
    fg_box = _bbox_from_binary(binary, (0, 0))
    if fg_box is None:
        return None
    _, _, fg_w, _ = fg_box
    best = max(candidates, key=lambda line: line.bbox[2])
    return best if best.bbox[2] >= fg_w * 0.45 else None


def _detect_full_sqrt(binary: np.ndarray) -> LineCandidate | None:
    candidates = [line for line in _line_candidates(binary) if _as_sqrt_line(binary, line) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda line: line.bbox[2])


def _is_fraction_line(binary: np.ndarray, line: LineCandidate) -> bool:
    x, y, w, h = line.bbox
    if line.source == "projection" and (binary.shape[0] < 90 or w < 65 or h > 8):
        return False
    x0, x1 = _content_x_range_near_line(binary, line)
    above = (binary[:y, x0:x1] < 255).sum()
    below = (binary[y + h :, x0:x1] < 255).sum()
    return above > 12 and below > 12 and w >= 28


def _as_sqrt_line(binary: np.ndarray, line: LineCandidate) -> LineCandidate | None:
    x, y, w, h = line.bbox
    if line.source != "projection" or w < 45:
        return None
    above = (binary[:y, max(0, x - 5): min(binary.shape[1], x + w + 5)] < 255).sum()
    below = (binary[y + h :, x: min(binary.shape[1], x + w + 4)] < 255).sum()
    left = _sqrt_left_edge(binary, line)
    radical_width = x - left
    radical_pixels = (binary[:, left:x + 3] < 255).sum()
    if above <= 8 and below > 12 and radical_pixels > 14 and 8 <= radical_width <= 70:
        return line
    return None


def _sqrt_left_edge(binary: np.ndarray, line: LineCandidate) -> int:
    x, y, _, _ = line.bbox
    search_x0 = max(0, x - 55)
    search = binary[:, search_x0:x]
    cols = np.where((search < 255).any(axis=0))[0]
    if cols.size == 0:
        return x
    return max(0, search_x0 + int(cols[0]) - 8)


def _content_x_range_near_line(binary: np.ndarray, line: LineCandidate) -> tuple[int, int]:
    x, y, w, h = line.bbox
    x0 = max(0, x - 18)
    x1 = min(binary.shape[1], x + w + 18)
    mask = binary < 255
    cols = np.where(mask[:, x0:x1].any(axis=0))[0]
    if cols.size == 0:
        return x0, x1
    return x0 + int(cols[0]), x0 + int(cols[-1]) + 1


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


def _meaningful_component_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = binary.shape[:2]
    max_area = int(height * width * 0.45)
    return [
        box
        for box in _component_boxes(binary)
        if box[2] * box[3] <= max_area and box[2] <= width * 0.75 and box[3] <= height * 0.75
    ]


def _has_script_layout(boxes: list[tuple[int, int, int, int]]) -> bool:
    for index, base in enumerate(boxes):
        bx, by, bw, bh = base
        if bh < 6 or bw < 3:
            continue
        for candidate_index, candidate in enumerate(boxes):
            if index == candidate_index:
                continue
            cx, cy, cw, ch = candidate
            if ch < 3 or cw < 2:
                continue
            if cx < bx + bw * 0.45:
                continue
            if cx > bx + bw + max(bh, bw) * 0.35:
                continue
            if ch < bh * 0.88 and cy + ch < by + bh * 0.55:
                return True
            if ch < bh * 0.88 and cy > by + bh * 0.45:
                return True
    return False


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


def _geometry_override(image: np.ndarray, label: str, score: float) -> str:
    foreground = image < 255
    h, w = image.shape[:2]
    area = int(foreground.sum())
    if area == 0:
        return label
    if w <= 10 and h <= 10:
        return "."
    row_counts = foreground.sum(axis=1)
    active_rows = np.where(row_counts > max(1, int(w * 0.35)))[0]
    if active_rows.size > 0:
        runs = _runs(active_rows)
        if len(runs) >= 2 and w / max(1, h) > 1.2:
            return "="
        if len(runs) == 1 and w / max(1, h) > 3:
            return "-"
    return label


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    start = int(values[0])
    previous = int(values[0])
    runs: list[tuple[int, int]] = []
    for value in map(int, values[1:]):
        if value - previous > 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _crop_binary(binary: np.ndarray) -> np.ndarray:
    x, y, w, h = foreground_bbox(binary)
    return binary[y:y + h, x:x + w]


def _foreground_offset(binary: np.ndarray) -> tuple[int, int]:
    x, y, _, _ = foreground_bbox(binary)
    return x, y


def _is_empty(binary: np.ndarray) -> bool:
    return binary.size == 0 or not (binary < 255).any()


def _bbox_from_binary(binary: np.ndarray, offset: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if _is_empty(binary):
        return None
    x, y, w, h = foreground_bbox(binary)
    return offset[0] + x, offset[1] + y, w, h


def _union_bbox(boxes: list[tuple[int, int, int, int] | None]) -> tuple[int, int, int, int] | None:
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    x0 = min(box[0] for box in valid)
    y0 = min(box[1] for box in valid)
    x1 = max(box[0] + box[2] for box in valid)
    y1 = max(box[1] + box[3] for box in valid)
    return x0, y0, x1 - x0, y1 - y0


def _overlaps_any(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(_iou(box, other) > 0.05 for other in boxes)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / max(1, union)


def _kind_for_label(label: str) -> str:
    if label.isdigit():
        return "digit"
    if label in {"+", "-", "÷", "=", "^"}:
        return "operator"
    if label in {"(", ")"}:
        return "paren"
    return "symbol"
