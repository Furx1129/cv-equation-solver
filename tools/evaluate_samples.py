from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.expression.display_normalizer import normalize_for_comparison, normalize_for_display
from src.expression.serializer import normalize_expression_text_for_sympy
from src.expression.types import ExpressionResult, RecognitionResult, SymbolToken
from src.solver.symbolic import SymbolicSolver
from src.vision.auto_router import format_candidate_scores, format_reject_info, recognize_unknown
from src.vision.layout_analysis import analyze_layout
from src.vision.calculus_layout import recognize_calculus_layout
from src.vision.normalization import normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image
from src.vision.recognizers.handwritten_rule_template import (
    DEFAULT_HANDWRITTEN_LABEL_DIR,
    DEFAULT_HANDWRITTEN_SAMPLE_DIR,
    load_handwritten_templates,
    match_handwritten_symbol,
)
from src.vision.recognizers.printed_template import kind_for_label
from src.vision.template_matcher import extract_geometry_features
from src.vision.pipeline import recognize_image
from src.vision.segmentation import segment_characters
from src.vision.structure_2d import recognize_2d_layout


SAMPLE_ROOT = PROJECT_ROOT / "data" / "samples"
LABEL_ROOT = PROJECT_ROOT / "data" / "labels"
REPORT_ROOT = PROJECT_ROOT / "reports" / "evaluation"
DEFAULT_CATEGORIES = (
    "printed_basic",
    "printed_decimal_negative",
    "printed_2d_layout",
    "handwritten_basic",
    "calculus",
)
FIELDNAMES = [
    "category",
    "image",
    "augmentation",
    "label",
    "predicted",
    "match",
    "selected_category",
    "route_hint",
    "route_confidence",
    "component_count",
    "line_count",
    "fraction_line_count",
    "foreground_fill_ratio",
    "candidate_scores",
    "router_reason",
    "reject_info",
    "layout_type",
    "sympy_text",
    "solver_answer",
    "solver_error",
    "token_count",
    "low_confidence_count",
    "unknown_count",
    "failure_stage",
    "failure_reason",
    "warnings",
    "top2_labels",
    "top2_scores",
    "decision_type",
    "num_holes",
    "aspect_ratio",
]


@dataclass(frozen=True)
class EvaluationRow:
    category: str
    image: str
    augmentation: str
    label: str
    predicted: str
    match: bool
    selected_category: str
    route_hint: str
    route_confidence: str
    component_count: str
    line_count: str
    fraction_line_count: str
    foreground_fill_ratio: str
    candidate_scores: str
    router_reason: str
    reject_info: str
    layout_type: str
    sympy_text: str
    solver_answer: str
    solver_error: str
    token_count: int
    low_confidence_count: int
    unknown_count: int
    failure_stage: str
    failure_reason: str
    warnings: str
    top2_labels: str = ""
    top2_scores: str = ""
    decision_type: str = ""
    num_holes: str = ""
    aspect_ratio: str = ""

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sample recognition against data/labels.")
    parser.add_argument("--category", default="all", choices=["all", "auto", *DEFAULT_CATEGORIES])
    parser.add_argument("--output", default=str(REPORT_ROOT / "evaluation_results.csv"))
    parser.add_argument("--debug-failures", action="store_true")
    parser.add_argument("--disable-fallbacks", action="store_true")
    parser.add_argument(
        "--augment-morphology",
        action="store_true",
        help="Also evaluate eroded and dilated copies to test morphology robustness.",
    )
    parser.add_argument("--solver-timeout-ms", type=int, default=1000)
    parser.add_argument("--confusion-matrix", action="store_true", help="Output confusion matrix to stdout and CSV")
    args = parser.parse_args()

    auto_mode = args.category == "auto"
    categories = DEFAULT_CATEGORIES if args.category in {"all", "auto"} else (args.category,)
    augmentations = ("none", "erode", "dilate") if args.augment_morphology else ("none",)
    rows: list[EvaluationRow] = []
    for category in categories:
        rows.extend(
            evaluate_category(
                category,
                debug_failures=args.debug_failures,
                auto_mode=auto_mode,
                disable_fallbacks=args.disable_fallbacks,
                solver_timeout_ms=args.solver_timeout_ms,
                augmentations=augmentations,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())

    _print_summary(rows)
    if args.confusion_matrix:
        _print_confusion_matrix(rows)
        _write_confusion_matrix_csv(rows, output.with_name(f"{output.stem}.confusion_matrix.csv"))
    return 0


def evaluate_category(
    category: str,
    debug_failures: bool = False,
    auto_mode: bool = False,
    disable_fallbacks: bool = False,
    solver_timeout_ms: int = 1000,
    augmentations: tuple[str, ...] = ("none",),
) -> list[EvaluationRow]:
    sample_dir = SAMPLE_ROOT / category
    label_dir = LABEL_ROOT / category
    if not sample_dir.exists():
        raise FileNotFoundError(f"sample category not found: {sample_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label category not found: {label_dir}")

    rows: list[EvaluationRow] = []
    for image_path in _iter_images(sample_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for {image_path.name}: {label_path}")
        label = label_path.read_text(encoding="utf-8").strip()
        for augmentation in augmentations:
            eval_image = _prepare_augmented_image(image_path, category, augmentation)
            debug_dir = None
            if debug_failures:
                debug_dir = PROJECT_ROOT / "debug" / "evaluation" / category / image_path.stem / augmentation
            rows.append(
                evaluate_one(
                    category,
                    eval_image,
                    label,
                    debug_dir=debug_dir,
                    auto_mode=auto_mode,
                    disable_fallbacks=disable_fallbacks,
                    solver_timeout_ms=solver_timeout_ms,
                    display_image_name=image_path.name,
                    augmentation=augmentation,
                )
            )
    return rows


def evaluate_one(
    category: str,
    image_path: Path,
    label: str,
    debug_dir: Path | None = None,
    auto_mode: bool = False,
    disable_fallbacks: bool = False,
    solver_timeout_ms: int = 1000,
    display_image_name: str | None = None,
    augmentation: str = "none",
) -> EvaluationRow:
    try:
        recognition, diag = _recognize_for_category(
            category,
            image_path,
            debug_dir=debug_dir,
            auto_mode=auto_mode,
            disable_fallbacks=disable_fallbacks,
            solver_timeout_ms=solver_timeout_ms,
        )
        predicted = recognition.expression_text
        label_key = _comparison_key(label, category)
        predicted_key = _comparison_key(predicted, category, recognition=recognition)
        matched = label_key == predicted_key
        display_label = normalize_for_display(label, category)
        display_predicted = normalize_for_display(predicted, category)
        sympy_text = _sympy_text(predicted, category, recognition)
        solve_answer, solve_error = _solve_if_symbolic(category, sympy_text, recognition)
        low_conf = sum(1 for token in recognition.tokens if token.confidence < 0.55)
        unknown = sum(1 for token in recognition.tokens if token.text == "UNKNOWN")
        stage, reason = _classify_failure(recognition, matched, solve_error, label_key, predicted_key)
        return EvaluationRow(
            category=category,
            image=display_image_name or image_path.name,
            augmentation=augmentation,
            label=display_label,
            predicted=display_predicted,
            match=matched,
            selected_category=diag.get("selected_category", ""),
            route_hint=diag.get("route_hint", ""),
            route_confidence=diag.get("route_confidence", ""),
            component_count=diag.get("component_count", ""),
            line_count=diag.get("line_count", ""),
            fraction_line_count=diag.get("fraction_line_count", ""),
            foreground_fill_ratio=diag.get("foreground_fill_ratio", ""),
            candidate_scores=diag.get("candidate_scores", ""),
            router_reason=diag.get("router_reason", ""),
            reject_info=diag.get("reject_info", ""),
            layout_type=recognition.layout.node_type if recognition.layout is not None else "",
            sympy_text=sympy_text,
            solver_answer=solve_answer,
            solver_error=solve_error,
            token_count=len(recognition.tokens),
            low_confidence_count=low_conf,
            unknown_count=unknown,
            failure_stage=stage,
            failure_reason=reason,
            warnings=" | ".join(recognition.warnings),
            top2_labels=diag.get("top2_labels", ""),
            top2_scores=diag.get("top2_scores", ""),
            decision_type=diag.get("decision_type", ""),
            num_holes=diag.get("num_holes", ""),
            aspect_ratio=diag.get("aspect_ratio", ""),
        )
    except Exception as exc:
        return EvaluationRow(
            category=category,
            image=display_image_name or image_path.name,
            augmentation=augmentation,
            label=label,
            predicted="",
            match=False,
            selected_category="",
            route_hint="",
            route_confidence="",
            component_count="",
            line_count="",
            fraction_line_count="",
            foreground_fill_ratio="",
            candidate_scores="",
            router_reason="",
            reject_info="",
            layout_type="",
            sympy_text="",
            solver_answer="",
            solver_error="",
            token_count=0,
            low_confidence_count=0,
            unknown_count=0,
            failure_stage="segmentation",
            failure_reason=str(exc),
            warnings="",
        )


def _recognize_for_category(
    category: str,
    image_path: Path,
    debug_dir: Path | None = None,
    auto_mode: bool = False,
    disable_fallbacks: bool = False,
    solver_timeout_ms: int = 1000,
) -> tuple[RecognitionResult, dict[str, object]]:
    if auto_mode:
        decision = recognize_unknown(
            image_path=image_path,
            debug_dir=debug_dir,
            solver_timeout_ms=solver_timeout_ms,
            disable_fallbacks=disable_fallbacks,
        )
        return decision.selected.result or RecognitionResult(tokens=[], expression_text=""), {
            "selected_category": decision.selected.category,
            "route_hint": decision.structure_analysis.route_hint,
            "route_confidence": f"{decision.structure_analysis.route_confidence:.3f}",
            "component_count": str(decision.image_features.foreground_components),
            "line_count": str(decision.image_features.long_horizontal_lines),
            "fraction_line_count": str(decision.image_features.fraction_like_lines),
            "foreground_fill_ratio": f"{decision.image_features.foreground_fill_ratio:.4f}",
            "candidate_scores": format_candidate_scores(decision),
            "router_reason": decision.router_reason,
            "reject_info": format_reject_info(decision),
        }
    if category == "handwritten_basic":
        return _recognize_handwritten_leave_one_out(image_path)
    if category == "calculus":
        return recognize_calculus_layout(
            image_path=image_path,
            debug_dir=debug_dir,
            disable_fallbacks=disable_fallbacks,
        ), {}
    if category == "printed_2d_layout":
        return recognize_2d_layout(image_path=image_path, debug_dir=debug_dir), {}
    return recognize_image(image_path=image_path, debug_dir=debug_dir), {}


def _recognize_handwritten_leave_one_out(image_path: Path) -> tuple[RecognitionResult, dict[str, object]]:
    templates = load_handwritten_templates(
        DEFAULT_HANDWRITTEN_SAMPLE_DIR,
        DEFAULT_HANDWRITTEN_LABEL_DIR,
        exclude_stem=image_path.stem,
    )
    image = read_image(image_path)
    normalized = normalize_formula_image(image)
    preprocessed = preprocess_image(
        normalized.image,
        options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
    )
    segments = segment_characters(preprocessed.binary)
    tokens: list[SymbolToken] = []
    warnings: list[str] = []
    diag: dict[str, object] = {}
    for idx, segment in enumerate(segments):
        label, score, candidates, decision_type = match_handwritten_symbol(segment.image, templates)
        tokens.append(
            SymbolToken(
                text=label,
                kind=kind_for_label(label),
                bbox=segment.bbox,
                confidence=score,
                source="handwritten_leave_one_out",
                candidates=candidates,
            )
        )
        if score < 0.45:
            warnings.append(f"low confidence handwritten token: {label} score={score:.3f}")
        # Per-token diagnostics: first token is enough for single-symbol evaluation.
        if idx == 0:
            top_items = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
            if len(top_items) >= 2:
                diag["top2_labels"] = f"{top_items[0][0]},{top_items[1][0]}"
                diag["top2_scores"] = f"{top_items[0][1]:.4f},{top_items[1][1]:.4f}"
            diag["decision_type"] = decision_type
            feat = extract_geometry_features(segment.image)
            diag["num_holes"] = str(int(feat["num_holes"]))
            diag["aspect_ratio"] = str(feat["aspect_ratio"])
    layout = analyze_layout(tokens)
    expression = "".join(token.text for token in tokens)
    return RecognitionResult(
        tokens=tokens,
        expression_text=expression,
        warnings=warnings,
        layout=layout,
        sympy_text=normalize_expression_text_for_sympy(expression),
    ), diag


def _comparison_key(label: str, category: str, recognition: RecognitionResult | None = None) -> str:
    if category == "calculus":
        if recognition is not None and recognition.sympy_text:
            return _compact_sympy(recognition.sympy_text)
        return _compact_sympy(normalize_expression_text_for_sympy(label))
    return normalize_for_comparison(label, category)


def _sympy_text(label_or_prediction: str, category: str, recognition: RecognitionResult) -> str:
    if category != "calculus":
        return recognition.sympy_text or ""
    if recognition.sympy_text:
        return recognition.sympy_text
    return normalize_expression_text_for_sympy(label_or_prediction)


def _solve_if_symbolic(
    category: str,
    sympy_text: str,
    recognition: RecognitionResult,
) -> tuple[str, str]:
    if category != "calculus" or not sympy_text:
        return "", ""
    result = SymbolicSolver().solve(ExpressionResult(text=sympy_text))
    return str(result.answer) if result.answer is not None else "", result.error or ""


def _classify_failure(
    recognition: RecognitionResult,
    matched: bool,
    solve_error: str,
    label_key: str,
    predicted_key: str,
) -> tuple[str, str]:
    if matched and not solve_error:
        return "", ""
    if not recognition.tokens:
        return "segmentation", "no tokens produced"
    if any(token.text == "UNKNOWN" for token in recognition.tokens):
        return "symbol", "unknown symbol token"
    if any(token.confidence < 0.55 for token in recognition.tokens):
        return "symbol", "low confidence token"
    if label_key != predicted_key:
        return "layout", f"expected {label_key}, got {predicted_key}"
    if solve_error:
        return "solver", solve_error
    return "serialization", "unmatched expression after normalization"


def _compact_sympy(text: str) -> str:
    return "".join(text.replace("^", "**").split())


def _iter_images(sample_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _print_summary(rows: list[EvaluationRow]) -> None:
    print(f"wrote {len(rows)} rows")
    for category in sorted({row.category for row in rows}):
        subset = [row for row in rows if row.category == category]
        correct = sum(1 for row in subset if row.match)
        print(f"{category}: {correct}/{len(subset)}")


def _print_confusion_matrix(rows: list[EvaluationRow]) -> None:
    true_categories = sorted({row.category for row in rows})
    pred_categories = sorted({row.selected_category or "unknown" for row in rows})
    columns = sorted(set(true_categories) | set(pred_categories))
    if not columns:
        return

    col_width = max(12, max(len(c) for c in columns))
    header_label = "true\\pred"
    header = f"{header_label:<{col_width}}" + "".join(f"{c:>{col_width}}" for c in columns)
    print("\nConfusion Matrix:")
    print(header)
    print("-" * len(header))

    matrix: dict[tuple[str, str], int] = {}
    for row in rows:
        true = row.category
        pred = row.selected_category or "unknown"
        key = (true, pred)
        matrix[key] = matrix.get(key, 0) + 1

    for true_cat in true_categories:
        line = f"{true_cat:<{col_width}}"
        total = sum(matrix.get((true_cat, pc), 0) for pc in columns)
        for pred_cat in columns:
            count = matrix.get((true_cat, pred_cat), 0)
            line += f"{count:>{col_width}}"
        line += f"  (n={total})"
        print(line)

    correct = sum(row.match for row in rows)
    total = len(rows)
    print(f"\nOverall: {correct}/{total} = {correct / max(1, total) * 100:.1f}%")


def _write_confusion_matrix_csv(rows: list[EvaluationRow], csv_path: Path) -> None:
    true_categories = sorted({row.category for row in rows})
    pred_categories = sorted({row.selected_category or "unknown" for row in rows})
    columns = sorted(set(true_categories) | set(pred_categories))

    matrix: dict[tuple[str, str], int] = {}
    for row in rows:
        true = row.category
        pred = row.selected_category or "unknown"
        key = (true, pred)
        matrix[key] = matrix.get(key, 0) + 1

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + columns)
        for true_cat in true_categories:
            writer.writerow([true_cat] + [matrix.get((true_cat, pc), 0) for pc in columns])

    print(f"Confusion matrix CSV written to {csv_path}")


def _prepare_augmented_image(image_path: Path, category: str, augmentation: str) -> Path:
    if augmentation == "none":
        return image_path
    if augmentation not in {"erode", "dilate"}:
        raise ValueError(f"unknown augmentation: {augmentation}")

    output = PROJECT_ROOT / "debug" / "augmentation" / augmentation / category / image_path.name
    output.parent.mkdir(parents=True, exist_ok=True)

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = 255 - binary
    kernel = np.ones((2, 2), dtype=np.uint8)
    if augmentation == "erode":
        changed = cv2.erode(foreground, kernel, iterations=1)
    else:
        changed = cv2.dilate(foreground, kernel, iterations=1)
    augmented = 255 - changed
    ok, encoded = cv2.imencode(output.suffix or ".png", augmented)
    if not ok:
        raise OSError(f"failed to encode augmented image: {output}")
    encoded.tofile(str(output))
    return output


if __name__ == "__main__":
    raise SystemExit(main())
