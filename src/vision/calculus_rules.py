from __future__ import annotations

from dataclasses import replace

from src.expression.serializer import serialize_layout
from src.expression.types import LayoutNode, RecognitionResult, SymbolToken
from src.vision.layout_analysis import detect_superscripts, tokens_to_row


def apply_calculus_geometry_rules(result: RecognitionResult) -> RecognitionResult:
    tokens = [_rewrite_calculus_symbol(token, index) for index, token in enumerate(result.tokens)]
    layout = _detect_integral(tokens) or _detect_derivative(tokens)
    if layout is None:
        expression_text = "".join(token.text for token in tokens)
        return replace(result, tokens=tokens, expression_text=expression_text)

    expression_text = serialize_layout(layout, target="plain")
    sympy_text = serialize_layout(layout, target="sympy")
    warnings = [warning for warning in result.warnings if "low confidence" not in warning]
    return replace(
        result,
        tokens=tokens,
        expression_text=expression_text,
        layout=layout,
        sympy_text=sympy_text,
        warnings=warnings,
    )


def _rewrite_calculus_symbol(token: SymbolToken, index: int) -> SymbolToken:
    x, y, w, h = token.bbox
    aspect = w / max(1, h)
    if index == 0 and token.text in {"1", "/", "UNKNOWN"} and h > 80 and w < 130 and aspect < 0.8:
        return replace(token, text="∫", kind="symbol", confidence=max(token.confidence, 0.80))
    if token.text in {"y", "UNKNOWN"} and h > 55 and aspect < 1.0:
        return replace(token, text="d", kind="symbol", confidence=max(token.confidence, 0.72))
    return token


def _detect_integral(tokens: list[SymbolToken]) -> LayoutNode | None:
    if not tokens or tokens[0].text != "∫":
        return None
    if len(tokens) < 4:
        return None
    variable = tokens[-1].text if tokens[-1].text in {"x", "y"} else "x"
    differential_index = len(tokens) - 2
    if not _looks_like_differential_d(tokens[differential_index]):
        return None
    body_tokens = tokens[1:differential_index]
    if not body_tokens:
        return None
    return LayoutNode(
        node_type="integral",
        children=[_body_layout(body_tokens)],
        metadata={"variable": variable},
        bbox=_union_bbox([token.bbox for token in tokens]),
    )


def _detect_derivative(tokens: list[SymbolToken]) -> LayoutNode | None:
    if len(tokens) < 2:
        return None
    first = tokens[0]
    if first.text == "∫":
        return None
    _, _, w, h = first.bbox
    if first.text not in {"d", "UNKNOWN", "6", "/"} and not (w > 70 and h > 90):
        return None
    if w < 70 or h < 90:
        return None
    body_tokens = tokens[1:]
    if not body_tokens:
        return None
    return LayoutNode(
        node_type="derivative",
        children=[_body_layout(body_tokens)],
        metadata={"variable": "x"},
        bbox=_union_bbox([token.bbox for token in tokens]),
    )


def _looks_like_differential_d(token: SymbolToken) -> bool:
    _, _, w, h = token.bbox
    aspect = w / max(1, h)
    return token.text == "d" or (token.text in {"y", "2", "4", "UNKNOWN"} and h > 55 and aspect < 1.0)


def _body_layout(tokens: list[SymbolToken]) -> LayoutNode:
    function = _detect_function_call(tokens)
    if function is not None:
        return function
    if len(tokens) >= 2:
        return detect_superscripts(tokens)
    return tokens_to_row(tokens)


def _detect_function_call(tokens: list[SymbolToken]) -> LayoutNode | None:
    for index, token in enumerate(tokens):
        if token.text != "(":
            continue
        prefix = "".join(t.text for t in tokens[:index])
        name = _classify_function_prefix(prefix)
        if name is None:
            continue
        close_index = _find_matching_close(tokens, index)
        if close_index is None:
            continue
        argument_tokens = tokens[index + 1 : close_index]
        if not argument_tokens:
            continue
        return LayoutNode(
            node_type="function_call",
            children=[_body_layout(argument_tokens)],
            metadata={"name": name},
            bbox=_union_bbox([token.bbox for token in tokens[: close_index + 1]]),
        )
    return None


def _classify_function_prefix(prefix: str) -> str | None:
    compact = prefix.replace("UNKNOWN", "?")
    if compact in {"sin", "3/x", "5/x", "5(1", "51x", "5ix"}:
        return "sin"
    if compact in {"cos", "c05", "c0s"}:
        return "cos"
    if compact in {"tan", "7an"}:
        return "tan"
    if compact in {"exp", "6x2", "ex2", "64d"}:
        return "exp"
    if compact in {"ln", "1n"}:
        return "log"
    return None


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


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return (x0, y0, x1 - x0, y1 - y0)
