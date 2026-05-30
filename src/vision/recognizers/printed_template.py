from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.expression.normalizer import normalize_tokens
from src.expression.serializer import serialize_layout
from src.expression.types import RecognitionResult, SymbolToken
from src.vision.deskew import deskew_image
from src.vision.layout_analysis import analyze_layout
from src.vision.normalization import normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image, save_debug_image
from src.vision.segmentation import segment_characters
from src.vision.template_matcher import load_templates, match_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates" / "printed_basic"
ARITHMETIC_TEMPLATE_LABELS = frozenset("0123456789+-x÷()=.")
ARITHMETIC_TEMPLATE_EXCLUDE_FILENAMES = frozenset({"tem_mul_x.png"})


@dataclass(frozen=True)
class PrintedTemplateRecognizer:
    template_dir: Path = DEFAULT_TEMPLATE_DIR
    confidence_warning_threshold: float = 0.55
    normalize_input: bool = True
    deskew_input: bool = True
    template_labels: frozenset[str] | None = ARITHMETIC_TEMPLATE_LABELS
    exclude_template_filenames: frozenset[str] | None = ARITHMETIC_TEMPLATE_EXCLUDE_FILENAMES

    def recognize(self, image_path: str | Path, debug_dir: str | Path | None = None) -> RecognitionResult:
        templates = load_templates(
            Path(self.template_dir),
            include_labels=self.template_labels,
            exclude_filenames=self.exclude_template_filenames,
        )
        debug_base = Path(debug_dir) if debug_dir is not None else None
        debug_artifacts: dict[str, Path] = {}
        if debug_base is not None:
            debug_base.mkdir(parents=True, exist_ok=True)

        image = read_image(image_path)
        if self.normalize_input:
            normalized = normalize_formula_image(image)
            working = normalized.image
            if debug_base is not None:
                debug_artifacts["normalized"] = save_debug_image(debug_base / "normalized.png", working)
        else:
            working = image

        preprocessed = preprocess_image(
            working,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
        )
        binary = preprocessed.binary
        if self.deskew_input:
            deskewed = deskew_image(binary)
            binary = deskewed.image
            if debug_base is not None:
                debug_artifacts["deskewed"] = save_debug_image(debug_base / "deskewed.png", binary)

        segments = segment_characters(binary)
        if debug_base is not None:
            debug_artifacts["binary"] = save_debug_image(debug_base / "binary.png", binary)
            (debug_base / "segments").mkdir(exist_ok=True)

        tokens: list[SymbolToken] = []
        warnings: list[str] = []
        for index, segment in enumerate(segments):
            match = match_symbol(segment.image, templates)
            guessed = guess_geometric_symbol(segment.image, match.label, match.score)
            label = guessed if guessed is not None else match.label
            score = 0.80 if guessed is not None else match.score
            token = SymbolToken(
                text=label,
                kind=kind_for_label(label),
                bbox=segment.bbox,
                confidence=score,
                source="printed_template",
                candidates=dict(sorted(match.candidates.items())),
            )
            tokens.append(token)
            if score < self.confidence_warning_threshold:
                warnings.append(
                    f"low confidence token {index}: {label} score={score:.3f} bbox={segment.bbox}"
                )
            if debug_base is not None:
                save_debug_image(
                    debug_base / "segments" / f"segment_{index:02d}_{safe_label(label)}.png",
                    segment.image,
                )

        expression = normalize_tokens(tokens)
        layout = analyze_layout(tokens)
        warnings.extend(expression.warnings)
        if debug_base is not None:
            table_path = debug_base / "tokens.csv"
            with table_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "text", "kind", "bbox", "confidence", "source"])
                for index, token in enumerate(tokens):
                    writer.writerow([index, token.text, token.kind, token.bbox, f"{token.confidence:.6f}", token.source])
            debug_artifacts["tokens"] = table_path

        return RecognitionResult(
            tokens=tokens,
            expression_text=expression.text,
            debug_artifacts=debug_artifacts,
            warnings=warnings,
            layout=layout,
            sympy_text=serialize_layout(layout, target="sympy"),
        )


def kind_for_label(label: str) -> str:
    if label.isdigit():
        return "digit"
    if label in {"+", "-", "*", "/", "÷", "^", "=", "->", "→"}:
        return "operator"
    if label in {"(", ")"}:
        return "paren"
    if label in {".", "x", "y", "d", "∂", "∫", "√", "∞"}:
        return "symbol"
    if label.isalpha() or label in {"lim", "dx", "d/dx"}:
        return "symbol"
    return "unknown"


def safe_label(label: str) -> str:
    return {
        "*": "times",
        "/": "divide",
        "÷": "divide",
        "x": "x",
        "+": "plus",
        "-": "minus",
        "(": "left",
        ")": "right",
        ".": "dot",
        "=": "equal",
        "^": "caret",
        "√": "sqrt",
        "∫": "integral",
        "∂": "partial",
        "→": "arrow",
        "∞": "infinity",
        "UNKNOWN": "unknown",
    }.get(label, label)


def guess_geometric_symbol(image, matched_label: str, matched_score: float) -> str | None:
    if matched_score >= 0.45 and matched_label != "UNKNOWN":
        return None
    foreground = image < 255
    area = int(foreground.sum())
    h, w = image.shape[:2]
    if area == 0:
        return None
    if w <= 10 and h <= 10:
        return "."
    row_counts = foreground.sum(axis=1)
    active_rows = np.where(row_counts > max(1, int(w * 0.35)))[0]
    if active_rows.size == 0:
        return None
    runs = []
    start = int(active_rows[0])
    prev = int(active_rows[0])
    for row in map(int, active_rows[1:]):
        if row - prev > 1:
            runs.append((start, prev))
            start = row
        prev = row
    runs.append((start, prev))
    if len(runs) >= 2 and w / max(1, h) > 1.2:
        return "="
    if len(runs) == 1 and w / max(1, h) > 3.0:
        return "-"
    return None
