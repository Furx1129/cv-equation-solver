from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
import re

from src.expression.types import ExpressionResult, LayoutNode, RecognitionResult
from src.solver.arithmetic import ArithmeticSolver
from src.solver.symbolic import SymbolicSolver
from src.vision.calculus_layout import recognize_calculus_layout
from src.vision.recognizers.handwritten_rule_template import HandwrittenRuleTemplateRecognizer
from src.vision.recognizers.printed_template import PrintedTemplateRecognizer
from src.vision.structure_2d import recognize_2d_layout
from src.vision.structure_router import (
    AUTO_ROUTE_CATEGORIES,
    FormulaStructureAnalysis,
    ImageStructureFeatures,
    analyze_formula_structure,
    extract_image_structure_features,
)


AUTO_CATEGORIES = AUTO_ROUTE_CATEGORIES

_CONFIDENCE_BASELINE = {
    "printed_basic": (0.86, 0.12),
    "printed_decimal_negative": (0.84, 0.12),
    "printed_2d_layout": (0.80, 0.15),
    "handwritten_basic": (0.70, 0.18),
    "calculus": (0.72, 0.18),
}


@dataclass(frozen=True)
class RecognitionCandidate:
    category: str
    result: RecognitionResult | None
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reject_stage: str = ""
    reject_detail: str = ""
    solver_answer: str = ""
    solver_error: str = ""
    role: str = "candidate"

    @property
    def accepted(self) -> bool:
        return self.result is not None and not self.reject_stage


@dataclass(frozen=True)
class AutoRecognitionDecision:
    selected: RecognitionCandidate
    candidates: list[RecognitionCandidate]
    image_features: ImageStructureFeatures
    router_reason: str
    structure_analysis: FormulaStructureAnalysis


def recognize_unknown(
    image_path: str | Path,
    debug_dir: str | Path | None = None,
    solver_timeout_ms: int = 1000,
    disable_fallbacks: bool = False,
) -> AutoRecognitionDecision:
    analysis = analyze_formula_structure(image_path)
    candidates: list[RecognitionCandidate] = []
    run_categories: set[str] = set()

    primary = _run_candidate(
        category=analysis.route_hint,
        image_path=image_path,
        debug_dir=_candidate_debug_dir(debug_dir, analysis.route_hint),
        features=analysis.features,
        solver_timeout_ms=solver_timeout_ms,
        disable_fallbacks=disable_fallbacks,
        role="primary",
    )
    candidates.append(primary)
    run_categories.add(primary.category)

    narrow_margin = analysis.route_confidence <= 0.7

    if narrow_margin and not disable_fallbacks and len(analysis.fallback_categories) >= 1:
        second_category = analysis.fallback_categories[0]
        secondary = _run_candidate(
            category=second_category,
            image_path=image_path,
            debug_dir=_candidate_debug_dir(debug_dir, second_category),
            features=analysis.features,
            solver_timeout_ms=solver_timeout_ms,
            disable_fallbacks=disable_fallbacks,
            role="cross_validation",
        )
        candidates.append(secondary)
        run_categories.add(secondary.category)

        if secondary.result is not None and primary.result is not None:
            xv = _cross_validate(primary, secondary)
            primary_xv = RecognitionCandidate(
                category=primary.category, result=primary.result,
                score=primary.score + xv,
                score_breakdown={**primary.score_breakdown, "cross_validate": xv},
                reject_stage=primary.reject_stage, reject_detail=primary.reject_detail,
                solver_answer=primary.solver_answer, solver_error=primary.solver_error,
                role="primary",
            )
            secondary_xv = RecognitionCandidate(
                category=secondary.category, result=secondary.result,
                score=secondary.score + xv,
                score_breakdown={**secondary.score_breakdown, "cross_validate": xv},
                reject_stage=secondary.reject_stage, reject_detail=secondary.reject_detail,
                solver_answer=secondary.solver_answer, solver_error=secondary.solver_error,
                role="cross_validation",
            )
            candidates = [primary_xv, secondary_xv]
            run_categories = {primary.category, secondary.category}

            both_low = primary_xv.score < 0.0 and secondary_xv.score < 0.0
            if both_low:
                for fallback in analysis.fallback_categories[1:]:
                    if fallback not in run_categories:
                        fb = _run_candidate(
                            category=fallback, image_path=image_path,
                            debug_dir=_candidate_debug_dir(debug_dir, fallback),
                            features=analysis.features, solver_timeout_ms=solver_timeout_ms,
                            disable_fallbacks=disable_fallbacks, role="fallback",
                        )
                        candidates.append(fb)
                        run_categories.add(fb.category)
                        if fb.result is not None and fb.score >= 0.0:
                            break
        else:
            for fallback in analysis.fallback_categories[1:]:
                if fallback not in run_categories:
                    fb = _run_candidate(
                        category=fallback, image_path=image_path,
                        debug_dir=_candidate_debug_dir(debug_dir, fallback),
                        features=analysis.features, solver_timeout_ms=solver_timeout_ms,
                        disable_fallbacks=disable_fallbacks, role="fallback",
                    )
                    candidates.append(fb)
                    run_categories.add(fb.category)
                    if fb.result is not None and fb.score >= 0.0:
                        break
    elif _candidate_needs_fallback(primary):
        for fallback in analysis.fallback_categories:
            if fallback in run_categories:
                continue
            fallback_candidate = _run_candidate(
                category=fallback, image_path=image_path,
                debug_dir=_candidate_debug_dir(debug_dir, fallback),
                features=analysis.features, solver_timeout_ms=solver_timeout_ms,
                disable_fallbacks=disable_fallbacks, role="fallback",
            )
            candidates.append(fallback_candidate)
            run_categories.add(fallback_candidate.category)
            if not _candidate_needs_fallback(fallback_candidate):
                break

    for category in AUTO_CATEGORIES:
        if category not in {candidate.category for candidate in candidates}:
            candidates.append(
                RecognitionCandidate(
                    category=category,
                    result=None,
                    score=-999.0,
                    reject_stage="route_prior",
                    reject_detail=f"structure route selected {analysis.route_hint}",
                    role="skipped",
                )
            )

    runnable = [candidate for candidate in candidates if candidate.result is not None]
    if not runnable:
        selected = RecognitionCandidate(
            category="unknown",
            result=RecognitionResult(tokens=[], expression_text="", warnings=["auto router rejected all candidates"]),
            score=-999.0,
            reject_stage="routing",
            reject_detail="all candidates rejected by structure router",
            role="selected",
        )
        reason = _router_reason(selected, analysis)
        return AutoRecognitionDecision(
            selected=selected,
            candidates=candidates,
            image_features=analysis.features,
            router_reason=reason,
            structure_analysis=analysis,
        )

    selected = max(runnable, key=lambda candidate: _selection_score(candidate, analysis.route_hint))
    reason = _router_reason(selected, analysis)
    selected_result = _mark_auto_result(selected.result, selected.category, reason, analysis)
    selected = RecognitionCandidate(
        category=selected.category,
        result=selected_result,
        score=selected.score,
        score_breakdown=selected.score_breakdown,
        reject_stage=selected.reject_stage,
        reject_detail=selected.reject_detail,
        solver_answer=selected.solver_answer,
        solver_error=selected.solver_error,
        role="selected",
    )
    return AutoRecognitionDecision(
        selected=selected,
        candidates=candidates,
        image_features=analysis.features,
        router_reason=reason,
        structure_analysis=analysis,
    )


def format_candidate_scores(decision: AutoRecognitionDecision) -> str:
    parts = []
    for candidate in sorted(decision.candidates, key=lambda item: item.score, reverse=True):
        role = f":{candidate.role}" if candidate.role else ""
        if candidate.result is None:
            parts.append(f"{candidate.category}:reject:{candidate.reject_stage}{role}")
        else:
            parts.append(f"{candidate.category}:{candidate.score:.3f}{role}")
    return ";".join(parts)


def format_reject_info(decision: AutoRecognitionDecision) -> str:
    items = []
    for candidate in decision.candidates:
        if candidate.reject_stage:
            items.append(f"{candidate.category}:{candidate.reject_stage}:{candidate.reject_detail}")
    return " | ".join(items)


def _candidate_debug_dir(debug_dir: str | Path | None, category: str) -> Path | None:
    return Path(debug_dir) / category if debug_dir is not None else None


def _run_candidate(
    category: str,
    image_path: str | Path,
    debug_dir: Path | None,
    features: ImageStructureFeatures,
    solver_timeout_ms: int,
    disable_fallbacks: bool,
    role: str,
) -> RecognitionCandidate:
    try:
        result = _recognize_category(category, image_path, debug_dir, disable_fallbacks=disable_fallbacks)
    except Exception as exc:
        return RecognitionCandidate(
            category=category,
            result=None,
            score=-999.0,
            reject_stage="recognizer",
            reject_detail=str(exc),
            role=role,
        )

    category = _display_category_for_result(category, result)
    score, breakdown, stage, detail = _score_candidate(category, result, features)
    solver_answer, solver_error = _validate_semantics(category, result, solver_timeout_ms)
    if solver_error:
        breakdown["semantic"] = -0.25 if solver_error == "timeout" else -0.15
        score += breakdown["semantic"]
        if not stage:
            stage = "semantic_validation"
            detail = solver_error

    return RecognitionCandidate(
        category=category,
        result=result,
        score=score,
        score_breakdown=breakdown,
        reject_stage=stage,
        reject_detail=detail,
        solver_answer=solver_answer,
        solver_error=solver_error,
        role=role,
    )


def _recognize_category(
    category: str,
    image_path: str | Path,
    debug_dir: Path | None,
    disable_fallbacks: bool,
) -> RecognitionResult:
    if category in {"printed_basic", "printed_decimal_negative"}:
        return PrintedTemplateRecognizer().recognize(image_path=image_path, debug_dir=debug_dir)
    if category == "handwritten_basic":
        return HandwrittenRuleTemplateRecognizer().recognize(image_path=image_path, debug_dir=debug_dir)
    if category == "printed_2d_layout":
        return recognize_2d_layout(image_path=image_path, debug_dir=debug_dir)
    if category == "calculus":
        return recognize_calculus_layout(image_path=image_path, debug_dir=debug_dir, disable_fallbacks=disable_fallbacks)
    raise ValueError(f"unknown auto category: {category}")


def _display_category_for_result(category: str, result: RecognitionResult) -> str:
    if category not in {"printed_basic", "printed_decimal_negative"}:
        return category
    text = result.expression_text
    return "printed_decimal_negative" if "." in text or _has_unary_minus(text) else "printed_basic"


def _score_candidate(
    category: str,
    result: RecognitionResult,
    features: ImageStructureFeatures,
) -> tuple[float, dict[str, float], str, str]:
    token_count = len(result.tokens)
    unknown = sum(1 for token in result.tokens if token.text == "UNKNOWN")
    low_conf = sum(1 for token in result.tokens if token.confidence < 0.55)
    avg_conf = sum(token.confidence for token in result.tokens) / max(1, token_count)
    mean, spread = _CONFIDENCE_BASELINE[category]
    normalized_conf = (avg_conf - mean) / max(0.05, spread)
    ast_leaves = _layout_leaf_count(result.layout) if result.layout is not None else token_count
    complexity_ratio = min(ast_leaves, token_count) / max(1, features.foreground_components)

    breakdown: dict[str, float] = {
        "confidence": 0.25 * normalized_conf,
        "unknown": -0.55 * unknown,
        "low_confidence": -0.12 * low_conf,
        "complexity": _complexity_score(category, token_count, ast_leaves, complexity_ratio, features),
        "structure": _structure_score(category, result, features),
        "category_prior": _category_prior(category, result),
    }
    score = sum(breakdown.values())

    reject_stage = ""
    reject_detail = ""
    if token_count == 0 and result.layout is None:
        reject_stage = "recognition"
        reject_detail = "no tokens or layout returned"
        score -= 1.0
    elif unknown >= max(2, token_count // 2):
        reject_stage = "symbol"
        reject_detail = "too many UNKNOWN tokens"
        score -= 0.8
    elif (
        complexity_ratio < 0.18
        and features.foreground_components >= 8
        and not _is_valid_compact_layout(category, result.layout)
    ):
        reject_stage = "complexity_check"
        reject_detail = f"AST/tokens too small for {features.foreground_components} foreground components"
        score -= 0.8

    return score, breakdown, reject_stage, reject_detail


def _candidate_needs_fallback(candidate: RecognitionCandidate) -> bool:
    if candidate.result is None:
        return True
    if candidate.reject_stage:
        return True
    if candidate.score < 0.0:
        return True
    if candidate.score_breakdown.get("structure", 0.0) < -0.5:
        return True
    tokens = candidate.result.tokens
    if not tokens and candidate.result.layout is None:
        return True
    unknown = sum(1 for token in tokens if token.text == "UNKNOWN")
    low_conf = sum(1 for token in tokens if token.confidence < 0.50)
    avg_conf = sum(token.confidence for token in tokens) / max(1, len(tokens))
    return unknown > 0 or low_conf >= max(1, len(tokens) // 2) or avg_conf < 0.50


def _selection_score(candidate: RecognitionCandidate, route_hint: str) -> float:
    score = candidate.score
    if candidate.category == route_hint:
        score += 0.35
    if candidate.role == "primary":
        score += 0.15
    if candidate.solver_error:
        score -= 0.20
    return score


def _structure_score(category: str, result: RecognitionResult, features: ImageStructureFeatures) -> float:
    layout_type = result.layout.node_type if result.layout is not None else ""
    text = result.sympy_text or result.expression_text
    if category == "printed_2d_layout":
        layout_types = _layout_types(result.layout)
        if features.fraction_like_lines > 0 and "fraction" in layout_types:
            return 1.15
        if {"superscript", "subscript", "sqrt"} & layout_types:
            if _looks_like_numeric_script_artifact(result.expression_text):
                return -0.65
            return 1.15
        if features.fraction_like_lines == 0 and features.stacked_component_pairs == 0:
            return -0.80
        return -0.65
    if category == "calculus":
        if layout_type in {"limit", "integral", "derivative", "partial_derivative"}:
            return 0.85
        if any(name in text for name in ("integrate(", "diff(", "limit(")):
            return 0.45
        if features.fraction_like_lines == 0 and features.stacked_component_pairs == 0 and features.aspect_ratio > 4.5:
            return -0.45
        return -0.05
    if category == "handwritten_basic":
        if len(result.tokens) > 1:
            return -0.45
        if features.is_single_symbol_like:
            return 0.55
        if len(result.tokens) == 1 and features.foreground_components <= 4:
            return 0.25
        return -0.10
    if category in {"printed_basic", "printed_decimal_negative"}:
        if features.fraction_like_lines > 0:
            return -0.35
        return 0.15
    return 0.0


def _category_prior(category: str, result: RecognitionResult) -> float:
    text = result.expression_text
    if category == "printed_decimal_negative":
        return 0.15 if "." in text or _has_unary_minus(text) else -0.04
    if category == "printed_basic":
        return 0.06 if "." not in text and not _has_unary_minus(text) else -0.06
    return 0.0


def _complexity_score(
    category: str,
    token_count: int,
    ast_leaves: int,
    complexity_ratio: float,
    features: ImageStructureFeatures,
) -> float:
    if features.foreground_components <= 3:
        return 0.10 if token_count <= 3 else -0.20
    if category in {"printed_2d_layout", "calculus"} and ast_leaves >= 3:
        return min(0.25, complexity_ratio * 0.25)
    if token_count <= 1 and features.foreground_components >= 6:
        return -0.45
    if complexity_ratio < 0.25 and features.foreground_components >= 8:
        return -0.30
    return min(0.20, complexity_ratio * 0.20)


def _is_valid_compact_layout(category: str, layout: LayoutNode | None) -> bool:
    if category != "calculus" or layout is None:
        return False
    return layout.node_type in {"limit", "integral", "derivative", "partial_derivative"}


def _validate_semantics(
    category: str,
    result: RecognitionResult,
    timeout_ms: int,
) -> tuple[str, str]:
    if not result.expression_text and not result.sympy_text:
        return "", "empty expression"

    if category in {"printed_basic", "printed_decimal_negative"}:
        solve_result = _solve_with_timeout(ArithmeticSolver(), ExpressionResult(text=result.expression_text), timeout_ms)
    elif category == "printed_2d_layout":
        return "", ""
    else:
        text = result.sympy_text or result.expression_text
        solve_result = _solve_with_timeout(
            SymbolicSolver(),
            ExpressionResult(text=text, tokens=result.tokens, layout=result.layout),
            timeout_ms,
        )
    if solve_result is None:
        return "", "timeout"
    return str(solve_result.answer) if solve_result.answer is not None else "", solve_result.error or ""


def _solve_with_timeout(solver: object, expression: ExpressionResult, timeout_ms: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(solver.solve, expression)
    try:
        return future.result(timeout=timeout_ms / 1000.0)
    except TimeoutError:
        future.cancel()
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _layout_leaf_count(node: LayoutNode | None) -> int:
    if node is None:
        return 0
    if not node.children:
        return 1 if node.text or node.node_type == "symbol" else 0
    return sum(_layout_leaf_count(child) for child in node.children)


def _layout_types(node: LayoutNode | None) -> set[str]:
    if node is None:
        return set()
    types = {node.node_type}
    for child in node.children:
        types.update(_layout_types(child))
    return types


def _has_unary_minus(text: str) -> bool:
    stripped = text.replace(" ", "")
    return stripped.startswith("-") or "(-" in stripped or "=-" in stripped


def _looks_like_numeric_script_artifact(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(re.search(r"\d_|_\.", compact))


def _cross_validate(
    primary: RecognitionCandidate,
    secondary: RecognitionCandidate,
) -> float:
    agree = 0.0
    pr = primary.result
    sr = secondary.result
    if pr is None or sr is None:
        return agree
    p_set = {t.text for t in pr.tokens if t.text != "UNKNOWN"}
    s_set = {t.text for t in sr.tokens if t.text != "UNKNOWN"}
    if p_set and s_set:
        agree += (len(p_set & s_set) / len(p_set | s_set)) * 0.20
    if pr.layout is not None and sr.layout is not None:
        if pr.layout.node_type == sr.layout.node_type:
            agree += 0.05
    if primary.solver_answer and secondary.solver_answer:
        if primary.solver_answer == secondary.solver_answer:
            agree += 0.20
        else:
            agree -= 0.50
    return agree


def _router_reason(selected: RecognitionCandidate, analysis: FormulaStructureAnalysis) -> str:
    top_parts = sorted(selected.score_breakdown.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    parts = ", ".join(f"{name}={value:.2f}" for name, value in top_parts)
    return (
        f"route_hint={analysis.route_hint}; route_confidence={analysis.route_confidence:.2f}; "
        f"selected {selected.category}; score={selected.score:.3f}; {parts}; "
        f"reason={analysis.debug_reason}; features={analysis.features.as_dict()}"
    )


def _mark_auto_result(
    result: RecognitionResult | None,
    category: str,
    reason: str,
    analysis: FormulaStructureAnalysis,
) -> RecognitionResult:
    if result is None:
        return RecognitionResult(tokens=[], expression_text="", warnings=["auto router produced no result"])
    warnings = [
        *result.warnings,
        f"auto route hint: {analysis.route_hint} confidence={analysis.route_confidence:.2f}",
        f"auto selected category: {category}",
        f"auto router reason: {reason}",
    ]
    return RecognitionResult(
        tokens=result.tokens,
        expression_text=result.expression_text,
        debug_artifacts=result.debug_artifacts,
        warnings=warnings,
        layout=result.layout,
        sympy_text=result.sympy_text,
    )
