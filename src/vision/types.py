from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.expression.types import RecognitionResult


@dataclass(frozen=True)
class LineRegion:
    bbox: tuple[int, int, int, int]
    image: np.ndarray
    baseline_y: int | None = None


@dataclass(frozen=True)
class SymbolRegion:
    bbox: tuple[int, int, int, int]
    image: np.ndarray
    source: str
    area: int
    aspect_ratio: float
    baseline_position: str = "middle"
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class RecognizerBackend(Protocol):
    def recognize(self, image_path: str | Path, debug_dir: str | Path | None = None) -> RecognitionResult:
        ...
