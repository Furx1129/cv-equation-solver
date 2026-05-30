from __future__ import annotations

import re

from src.expression.types import LayoutNode


FUNCTION_NAMES = ("sin", "cos", "tan", "ln", "log", "exp", "sqrt")
_IDENT = r"[A-Za-z]"


def is_calculus_expression(text: str) -> bool:
    compact = _compact(text)
    return any(
        marker in compact
        for marker in ("∫", "int", "integral", "lim", "d/d", "∂/∂", "partial")
    )


def calculus_text_to_layout(text: str) -> LayoutNode | None:
    compact = _compact(text)
    compact = compact.replace("−", "-").replace("→", "->")
    compact = compact.replace("∞", "oo")

    layout = _parse_limit(compact)
    if layout is not None:
        return layout
    layout = _parse_integral(compact)
    if layout is not None:
        return layout
    layout = _parse_derivative(compact)
    if layout is not None:
        return layout
    return None


def calculus_text_to_sympy(text: str) -> str | None:
    layout = calculus_text_to_layout(text)
    if layout is None:
        return None
    from src.expression.serializer import serialize_layout

    return serialize_layout(layout, target="sympy")


def normalize_calculus_inner(text: str) -> str:
    compact = _compact(text)
    compact = compact.replace("−", "-").replace("→", "->").replace("∞", "oo")
    compact = compact.replace("^", "**").replace("×", "*").replace("÷", "/")
    compact = compact.replace("梅", "/").replace("脳", "*")
    compact = _normalize_ln(compact)
    compact = _normalize_exp_power(compact)
    return compact


def _parse_derivative(text: str) -> LayoutNode | None:
    patterns = [
        rf"^d/d(?P<var>{_IDENT})(?P<body>.+)$",
        rf"^∂/∂(?P<var>{_IDENT})(?P<body>.+)$",
        rf"^partial/partial(?P<var>{_IDENT})(?P<body>.+)$",
        rf"^d(?P<body>.+)/d(?P<var>{_IDENT})$",
        rf"^∂(?P<body>.+)/∂(?P<var>{_IDENT})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        body = _strip_wrapping_parens(match.group("body"))
        variable = match.group("var")
        return LayoutNode(
            node_type="derivative",
            children=[_text_node(body)],
            metadata={"variable": variable},
        )
    return None


def _parse_integral(text: str) -> LayoutNode | None:
    normalized = text
    for prefix in ("integral", "int", "∫"):
        if normalized.startswith(prefix):
            body = normalized[len(prefix) :]
            break
    else:
        return None

    lower = upper = None
    if body.startswith("_"):
        lower, body = _take_script_value(body[1:])
    if body.startswith("^"):
        upper, body = _take_script_value(body[1:])

    match = re.match(rf"^(?P<body>.+)d(?P<var>{_IDENT})$", body)
    if not match:
        return None
    integrand = _strip_wrapping_parens(match.group("body"))
    variable = match.group("var")
    metadata = {"variable": variable}
    if lower is not None and upper is not None:
        metadata.update({"lower": lower, "upper": upper})
    return LayoutNode(
        node_type="integral",
        children=[_text_node(integrand)],
        metadata=metadata,
    )


def _parse_limit(text: str) -> LayoutNode | None:
    if not text.startswith("lim"):
        return None
    body = text[3:]
    if body.startswith("_"):
        condition, body = _take_script_value(body[1:])
    else:
        match = re.match(rf"^(?P<var>{_IDENT})->(?P<target>[^A-Za-z]+)(?P<body>.+)$", body)
        if not match:
            return None
        condition = f"{match.group('var')}->{match.group('target')}"
        body = match.group("body")

    condition = condition.replace("to", "->")
    match = re.match(rf"^(?P<var>{_IDENT})->(?P<target>.+)$", condition)
    if not match:
        return None
    return LayoutNode(
        node_type="limit",
        children=[_text_node(_strip_wrapping_parens(body))],
        metadata={"variable": match.group("var"), "target": match.group("target")},
    )


def _take_script_value(text: str) -> tuple[str, str]:
    if text.startswith("{"):
        depth = 0
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[1:index], text[index + 1 :]
        return text[1:], ""
    if text.startswith("("):
        depth = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return text[1:index], text[index + 1 :]
        return text[1:], ""
    return text[0], text[1:]


def _text_node(text: str) -> LayoutNode:
    return LayoutNode(node_type="row", text=normalize_calculus_inner(text))


def _strip_wrapping_parens(text: str) -> str:
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1]
    return text


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _normalize_ln(text: str) -> str:
    return re.sub(r"\bln\(", "log(", text)


def _normalize_exp_power(text: str) -> str:
    return re.sub(r"\be\*\*(?P<body>[A-Za-z0-9_]+)", r"exp(\g<body>)", text)


def _compact(text: str) -> str:
    return "".join(text.strip().split())
