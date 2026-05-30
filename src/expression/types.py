from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class SymbolToken:
    text: str
    kind: str
    bbox: BBox
    confidence: float
    source: str
    candidates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RecognitionResult:
    tokens: list[SymbolToken]
    expression_text: str
    debug_artifacts: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    layout: "LayoutNode | None" = None
    sympy_text: str | None = None


@dataclass(frozen=True)
class ExpressionResult:
    text: str
    tokens: list[SymbolToken] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    layout: "LayoutNode | None" = None


@dataclass(frozen=True)
class SolveResult:
    answer: Any = None
    steps: list[str] = field(default_factory=list)
    backend: str = ""
    error: str | None = None


@dataclass(frozen=True)
class LayoutNode:
    node_type: str
    text: str | None = None
    children: list["LayoutNode"] = field(default_factory=list)
    bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def symbol(text: str, bbox: BBox | None = None) -> "LayoutNode":
        return LayoutNode(node_type="symbol", text=text, bbox=bbox)

    @staticmethod
    def row(children: list["LayoutNode"], bbox: BBox | None = None) -> "LayoutNode":
        return LayoutNode(node_type="row", children=children, bbox=bbox)
