from __future__ import annotations

import unicodedata


ARITHMETIC_CATEGORIES = {"printed_basic", "printed_decimal_negative"}


def normalize_for_display(text: str, category: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = normalized.replace("−", "-")
    if category in ARITHMETIC_CATEGORIES:
        return normalize_arithmetic_display(normalized)
    return normalized


def normalize_for_comparison(text: str, category: str) -> str:
    return "".join(normalize_for_display(text, category).split())


def normalize_arithmetic_display(text: str) -> str:
    result: list[str] = []
    for char in text:
        if char in {"×", "*", "脳"}:
            result.append("x")
        elif char in {"÷", "/", "梅"}:
            result.append("÷")
        else:
            result.append(char)
    return "".join(result)


def normalize_arithmetic_for_solver(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    result: list[str] = []
    for char in normalized:
        if char in {"x", "×", "*", "脳"}:
            result.append("*")
        elif char in {"÷", "/", "梅"}:
            result.append("/")
        elif char == "−":
            result.append("-")
        else:
            result.append(char)
    return "".join(result)
