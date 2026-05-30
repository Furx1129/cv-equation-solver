from __future__ import annotations

from .types import ExpressionResult, SymbolToken


SYMBOL_MAP = {
    "脳": "x",
    "梅": "÷",
    "×": "x",
    "÷": "÷",
    "−": "-",
    "X": "x",
    "√": "sqrt",
    "→": "->",
}


def normalize_tokens(tokens: list[SymbolToken]) -> ExpressionResult:
    warnings: list[str] = []
    chars: list[str] = []

    for index, token in enumerate(tokens):
        text = SYMBOL_MAP.get(token.text, token.text)
        if text == "UNKNOWN":
            warnings.append(f"token {index} is UNKNOWN at bbox={token.bbox}")
        chars.append(text)

    return ExpressionResult(text="".join(chars), tokens=tokens, warnings=warnings)
