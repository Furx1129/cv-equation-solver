from __future__ import annotations

from src.expression.calculus import calculus_text_to_sympy, normalize_calculus_inner
from src.expression.types import ExpressionResult, LayoutNode


def serialize_layout(node: LayoutNode, target: str = "plain") -> str:
    if node.node_type == "symbol":
        return _serialize_symbol(node.text or "", target=target)
    if node.node_type == "row":
        if node.text is not None:
            return normalize_calculus_inner(node.text) if target == "sympy" else node.text
        return "".join(serialize_layout(child, target=target) for child in node.children)
    if node.node_type == "fraction":
        numerator = serialize_layout(node.children[0], target=target)
        denominator = serialize_layout(node.children[1], target=target)
        if target == "plain":
            numerator = _wrap_fraction_part(numerator)
            denominator = _wrap_fraction_part(denominator)
            return f"{numerator}/{denominator}"
        numerator = _wrap_fraction_part(numerator)
        denominator = _wrap_fraction_part(denominator)
        return f"{numerator}/{denominator}"
    if node.node_type == "superscript":
        base = serialize_layout(node.children[0], target=target)
        exponent = serialize_layout(node.children[1], target=target)
        op = "**" if target == "sympy" else "^"
        return f"{base}{op}{exponent}"
    if node.node_type == "subscript":
        base = serialize_layout(node.children[0], target=target)
        sub = serialize_layout(node.children[1], target=target)
        return f"{base}_{sub}"
    if node.node_type == "sqrt":
        inner = serialize_layout(node.children[0], target=target)
        return f"sqrt({inner})"
    if node.node_type == "integral":
        body = serialize_layout(node.children[0], target=target) if node.children else (node.text or "")
        variable = node.metadata.get("variable", "x")
        lower = node.metadata.get("lower")
        upper = node.metadata.get("upper")
        if target == "sympy" and lower is not None and upper is not None:
            lower_text = normalize_calculus_inner(str(lower))
            upper_text = normalize_calculus_inner(str(upper))
            return f"integrate({body}, ({variable}, {lower_text}, {upper_text}))"
        return f"integrate({body}, {variable})"
    if node.node_type in {"derivative", "partial_derivative"}:
        body = serialize_layout(node.children[0], target=target) if node.children else (node.text or "")
        variable = node.metadata.get("variable", "x")
        return f"diff({body}, {variable})"
    if node.node_type == "limit":
        body = serialize_layout(node.children[0], target=target) if node.children else (node.text or "")
        variable = node.metadata.get("variable", "x")
        target_value = normalize_calculus_inner(str(node.metadata.get("target", "0")))
        return f"limit({body}, {variable}, {target_value})"
    if node.node_type == "function_call":
        name = node.metadata.get("name", node.text or "")
        args = ", ".join(serialize_layout(child, target=target) for child in node.children)
        if target == "sympy" and name == "ln":
            name = "log"
        return f"{name}({args})"
    return node.text or ""


def expression_to_sympy_text(expression: ExpressionResult) -> str:
    if expression.layout is not None:
        return serialize_layout(expression.layout, target="sympy")
    return normalize_expression_text_for_sympy(expression.text)


def normalize_expression_text_for_sympy(text: str) -> str:
    calculus = calculus_text_to_sympy(text)
    if calculus is not None:
        return calculus
    return normalize_calculus_inner(text)


def _serialize_symbol(text: str, target: str) -> str:
    if target == "sympy" and text == "^":
        return "**"
    if target == "sympy" and text in {"脳", "×"}:
        return "*"
    if target == "sympy" and text in {"梅", "÷"}:
        return "/"
    return text


def _wrap_fraction_part(text: str) -> str:
    if text.startswith("(") and text.endswith(")"):
        return text
    if any(op in text for op in ("+", "-", "=")):
        return f"({text})"
    return text
