from __future__ import annotations

from src.expression.calculus import calculus_text_to_layout
from src.expression.normalizer import normalize_tokens
from src.expression.types import LayoutNode, SymbolToken


def tokens_to_row(tokens: list[SymbolToken]) -> LayoutNode:
    children = [LayoutNode.symbol(token.text, bbox=token.bbox) for token in tokens]
    return LayoutNode.row(children=children, bbox=_union_bbox([token.bbox for token in tokens]))


def detect_simple_fraction(tokens: list[SymbolToken]) -> LayoutNode | None:
    if not tokens:
        return None
    line_candidates = [
        token
        for token in tokens
        if token.text in {"-", "/"} and token.bbox[2] > max(12, token.bbox[3] * 5)
    ]
    if not line_candidates:
        return None
    line = max(line_candidates, key=lambda token: token.bbox[2])
    lx, ly, lw, lh = line.bbox
    above = [token for token in tokens if token is not line and token.bbox[1] + token.bbox[3] <= ly + lh]
    below = [token for token in tokens if token is not line and token.bbox[1] >= ly]
    if not above or not below:
        return None
    return LayoutNode(
        node_type="fraction",
        children=[tokens_to_row(sorted(above, key=lambda t: t.bbox[0])), tokens_to_row(sorted(below, key=lambda t: t.bbox[0]))],
        bbox=_union_bbox([token.bbox for token in tokens]),
    )


def detect_superscripts(tokens: list[SymbolToken]) -> LayoutNode:
    if len(tokens) < 2:
        return tokens_to_row(tokens)
    sorted_tokens = sorted(tokens, key=lambda t: t.bbox[0])
    heights = [t.bbox[3] for t in sorted_tokens]
    median_height = sorted(heights)[len(heights) // 2]
    baseline = max(t.bbox[1] + t.bbox[3] for t in sorted_tokens)
    children: list[LayoutNode] = []
    i = 0
    while i < len(sorted_tokens):
        token = sorted_tokens[i]
        if i + 1 < len(sorted_tokens):
            nxt = sorted_tokens[i + 1]
            nxt_bottom = nxt.bbox[1] + nxt.bbox[3]
            if nxt.bbox[3] <= median_height * 0.75 and nxt_bottom < baseline - median_height * 0.25:
                children.append(
                    LayoutNode(
                        node_type="superscript",
                        children=[LayoutNode.symbol(token.text, token.bbox), LayoutNode.symbol(nxt.text, nxt.bbox)],
                        bbox=_union_bbox([token.bbox, nxt.bbox]),
                    )
                )
                i += 2
                continue
        children.append(LayoutNode.symbol(token.text, token.bbox))
        i += 1
    return LayoutNode.row(children=children, bbox=_union_bbox([t.bbox for t in sorted_tokens]))


def detect_sqrt(tokens: list[SymbolToken]) -> LayoutNode | None:
    if not tokens or tokens[0].text not in {"sqrt", "√"}:
        return None
    inner = tokens[1:]
    if not inner:
        return None
    return LayoutNode(
        node_type="sqrt",
        children=[tokens_to_row(sorted(inner, key=lambda t: t.bbox[0]))],
        bbox=_union_bbox([token.bbox for token in tokens]),
    )


def analyze_layout(tokens: list[SymbolToken]) -> LayoutNode:
    expression = normalize_tokens(tokens)
    calculus = calculus_text_to_layout(expression.text)
    if calculus is not None:
        return calculus
    sqrt = detect_sqrt(sorted(tokens, key=lambda t: t.bbox[0]))
    if sqrt is not None:
        return sqrt
    fraction = detect_simple_fraction(tokens)
    if fraction is not None:
        return fraction
    return detect_superscripts(tokens)


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return (x0, y0, x1 - x0, y1 - y0)
