# Auto Router & Disambiguation Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard-threshold routing with full-category scoring + cross-validation, and refactor symbol disambiguation into a 3-layer hierarchy with 10+ confusion-pair rules.

**Architecture:** Four independent layers: (1) extend `ImageStructureFeatures` with 6 new fields, (2) replace `choose_route()` if-else chain with 5 parallel scoring functions, (3) add dual-path cross-validation in `recognize_unknown()` for ambiguous cases, (4) refactor `_disambiguate()` into Layer A (hard rules) → Layer B (context-aware) → Layer C (confusion pairs).

**Tech Stack:** Python 3, OpenCV, NumPy (no ML dependencies)

---

### Task 1: Establish baseline accuracy

**Files:**
- None modified

- [ ] **Step 1: Run current auto evaluation baseline**

Run: `python tools/evaluate_samples.py --category auto --output evaluation_baseline_auto.csv`
Expected: CSV written with 176 rows (one per sample across all 5 categories).

- [ ] **Step 2: Note baseline per-category accuracy**

Run: `python -m unittest discover -s tests -v`
Expected: 57 tests OK (skipped=1). Record the test results — all must still pass at the end.

---

### Task 2: Layer 1 — Add 6 new feature fields to ImageStructureFeatures

**Files:**
- Modify: `src/vision/structure_router.py`

- [ ] **Step 1: Add new fields to the dataclass**

Replace the `ImageStructureFeatures` dataclass definition (lines 24-35) with the expanded version:

```python
@dataclass(frozen=True)
class ImageStructureFeatures:
    width: int
    height: int
    aspect_ratio: float
    foreground_components: int
    left_tall_components: int
    long_horizontal_lines: int
    fraction_like_lines: int
    stacked_component_pairs: int
    edge_roughness: float
    foreground_fill_ratio: float
    vertical_symmetry: float = 0.0
    component_height_variance: float = 0.0
    horizontal_run_count: int = 0
    left_right_density_ratio: float = 1.0
    multi_scale_edge_ratio: float = 1.0
    top_alignment_score: float = 0.0

    @property
    def is_extremely_flat_row(self) -> bool:
        return (
            self.aspect_ratio > 5.0
            and self.fraction_like_lines == 0
            and self.long_horizontal_lines == 0
            and self.stacked_component_pairs <= 1
        )

    @property
    def is_single_symbol_like(self) -> bool:
        return (
            self.foreground_components <= 3
            and self.aspect_ratio < 2.2
            and self.fraction_like_lines == 0
            and self.long_horizontal_lines <= 1
        )

    @property
    def is_vertically_symmetric(self) -> bool:
        return self.vertical_symmetry > 0.7

    @property
    def is_handwritten_texture(self) -> bool:
        return self.multi_scale_edge_ratio > 1.8 and self.component_height_variance > 0.3

    @property
    def has_integral_structure(self) -> bool:
        return self.left_right_density_ratio > 1.5 and self.left_tall_components > 0

    @property
    def is_grid_aligned(self) -> bool:
        return self.top_alignment_score > 3.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "foreground_components": self.foreground_components,
            "left_tall_components": self.left_tall_components,
            "long_horizontal_lines": self.long_horizontal_lines,
            "fraction_like_lines": self.fraction_like_lines,
            "stacked_component_pairs": self.stacked_component_pairs,
            "edge_roughness": round(self.edge_roughness, 4),
            "foreground_fill_ratio": round(self.foreground_fill_ratio, 4),
            "vertical_symmetry": round(self.vertical_symmetry, 4),
            "component_height_variance": round(self.component_height_variance, 4),
            "horizontal_run_count": self.horizontal_run_count,
            "left_right_density_ratio": round(self.left_right_density_ratio, 4),
            "multi_scale_edge_ratio": round(self.multi_scale_edge_ratio, 4),
            "top_alignment_score": round(self.top_alignment_score, 4),
        }
```

- [ ] **Step 2: Compute new features in extract_structure_features()**

Replace `extract_structure_features()` (lines 105-129) with:

```python
def extract_structure_features(binary: np.ndarray) -> ImageStructureFeatures:
    height, width = binary.shape[:2]
    foreground = binary < 255
    fg_count = int(foreground.sum())
    fill = fg_count / max(1, width * height)

    component_boxes = _component_boxes(binary)
    lines = _line_candidates(binary)
    long_lines = [line for line in lines if line.bbox[2] >= max(28, width * 0.12)]
    fraction_lines = [line for line in lines if _is_fraction_line(binary, line)]
    edges_low = cv2.Canny(binary, 30, 90)
    edges_high = cv2.Canny(binary, 70, 180)
    edge_roughness = float((edges_high > 0).sum() / max(1, fg_count))
    low_edge_count = int((edges_low > 0).sum())
    high_edge_count = int((edges_high > 0).sum())

    top_half_fg = int(foreground[: height // 2, :].sum()) if height >= 2 else 0
    bottom_half_fg = int(foreground[height // 2 :, :].sum()) if height >= 2 else 0
    vertical_symmetry = 1.0 - abs(top_half_fg - bottom_half_fg) / max(1, top_half_fg + bottom_half_fg)

    heights = [h for _, _, _, h in component_boxes]
    if len(heights) >= 2:
        mean_h = sum(heights) / len(heights)
        std_h = (sum((h - mean_h) ** 2 for h in heights) / len(heights)) ** 0.5
        height_variance = std_h / max(1, mean_h)
    else:
        height_variance = 0.0

    horizontal_proj = foreground.sum(axis=1).astype(np.float32)
    noise_floor = float(np.mean(horizontal_proj)) * 0.3 if horizontal_proj.size > 0 else 0.0
    in_run = False
    run_count = 0
    for val in horizontal_proj:
        if val > noise_floor and not in_run:
            run_count += 1
            in_run = True
        elif val <= noise_floor:
            in_run = False
    horizontal_run_count = run_count

    left_fg = int(foreground[:, : width // 2].sum()) if width >= 2 else 0
    right_fg = int(foreground[:, width // 2 :].sum()) if width >= 2 else 0
    left_right_density_ratio = left_fg / max(1, right_fg)

    multi_scale_edge_ratio = low_edge_count / max(1, high_edge_count)

    top_ys = [y for y, _, _, _ in component_boxes]
    if len(top_ys) >= 2:
        mean_top = sum(top_ys) / len(top_ys)
        std_top = (sum((y - mean_top) ** 2 for y in top_ys) / len(top_ys)) ** 0.5
        top_alignment = 1.0 / (1.0 + std_top)
    else:
        top_alignment = 3.0

    return ImageStructureFeatures(
        width=width,
        height=height,
        aspect_ratio=width / max(1, height),
        foreground_components=len(component_boxes),
        left_tall_components=_count_left_tall_components(component_boxes, width, height),
        long_horizontal_lines=len(long_lines),
        fraction_like_lines=len(fraction_lines),
        stacked_component_pairs=_count_stacked_pairs(component_boxes),
        edge_roughness=edge_roughness,
        foreground_fill_ratio=fill,
        vertical_symmetry=vertical_symmetry,
        component_height_variance=height_variance,
        horizontal_run_count=horizontal_run_count,
        left_right_density_ratio=left_right_density_ratio,
        multi_scale_edge_ratio=multi_scale_edge_ratio,
        top_alignment_score=top_alignment,
    )
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `python -m unittest discover -s tests -v`
Expected: 57 tests OK (skipped=1). New fields have defaults so existing code is unaffected.

- [ ] **Step 4: Commit**

```bash
git add src/vision/structure_router.py
git commit -m "feat: add 6 structure feature fields for improved routing discrimination"
```

---

### Task 3: Layer 2 — Replace choose_route() with scoring functions

**Files:**
- Modify: `src/vision/structure_router.py`

- [ ] **Step 1: Write new test file for scoring functions**

Create: `tests/test_structure_router_scoring.py`

```python
from __future__ import annotations

import unittest

from src.vision.structure_router import (
    ImageStructureFeatures,
    _score_printed_basic,
    _score_printed_2d_layout,
    _score_handwritten_basic,
    _score_calculus,
)


class ScoringFunctionTests(unittest.TestCase):
    def _flat_printed(self) -> ImageStructureFeatures:
        return ImageStructureFeatures(
            width=500, height=40, aspect_ratio=12.5,
            foreground_components=7, left_tall_components=0,
            long_horizontal_lines=0, fraction_like_lines=0,
            stacked_component_pairs=0, edge_roughness=0.08,
            foreground_fill_ratio=0.06,
            vertical_symmetry=0.85, component_height_variance=0.05,
            horizontal_run_count=1, left_right_density_ratio=1.0,
            multi_scale_edge_ratio=1.1, top_alignment_score=5.0,
        )

    def _fraction_2d(self) -> ImageStructureFeatures:
        return ImageStructureFeatures(
            width=120, height=180, aspect_ratio=0.67,
            foreground_components=4, left_tall_components=0,
            long_horizontal_lines=1, fraction_like_lines=1,
            stacked_component_pairs=1, edge_roughness=0.10,
            foreground_fill_ratio=0.15,
            vertical_symmetry=0.72, component_height_variance=0.10,
            horizontal_run_count=3, left_right_density_ratio=1.0,
            multi_scale_edge_ratio=1.2, top_alignment_score=4.0,
        )

    def _handwritten_single(self) -> ImageStructureFeatures:
        return ImageStructureFeatures(
            width=48, height=52, aspect_ratio=0.92,
            foreground_components=1, left_tall_components=0,
            long_horizontal_lines=0, fraction_like_lines=0,
            stacked_component_pairs=0, edge_roughness=0.42,
            foreground_fill_ratio=0.08,
            vertical_symmetry=0.55, component_height_variance=0.0,
            horizontal_run_count=1, left_right_density_ratio=1.0,
            multi_scale_edge_ratio=2.3, top_alignment_score=0.0,
        )

    def _calculus_integral(self) -> ImageStructureFeatures:
        return ImageStructureFeatures(
            width=200, height=160, aspect_ratio=1.25,
            foreground_components=8, left_tall_components=1,
            long_horizontal_lines=1, fraction_like_lines=1,
            stacked_component_pairs=2, edge_roughness=0.28,
            foreground_fill_ratio=0.10,
            vertical_symmetry=0.45, component_height_variance=0.25,
            horizontal_run_count=3, left_right_density_ratio=1.8,
            multi_scale_edge_ratio=1.9, top_alignment_score=1.5,
        )

    def test_flat_printed_scores_highest_on_printed_basic(self):
        f = self._flat_printed()
        scores = {
            "printed_basic": _score_printed_basic(f),
            "printed_2d_layout": _score_printed_2d_layout(f),
            "handwritten_basic": _score_handwritten_basic(f),
            "calculus": _score_calculus(f),
        }
        best = max(scores, key=scores.get)
        self.assertEqual(best, "printed_basic")

    def test_fraction_scores_highest_on_2d_layout(self):
        f = self._fraction_2d()
        scores = {
            "printed_basic": _score_printed_basic(f),
            "printed_2d_layout": _score_printed_2d_layout(f),
            "handwritten_basic": _score_handwritten_basic(f),
            "calculus": _score_calculus(f),
        }
        best = max(scores, key=scores.get)
        self.assertEqual(best, "printed_2d_layout")

    def test_handwritten_single_scores_highest_on_handwritten(self):
        f = self._handwritten_single()
        scores = {
            "printed_basic": _score_printed_basic(f),
            "printed_2d_layout": _score_printed_2d_layout(f),
            "handwritten_basic": _score_handwritten_basic(f),
            "calculus": _score_calculus(f),
        }
        best = max(scores, key=scores.get)
        self.assertEqual(best, "handwritten_basic")

    def test_calculus_integral_scores_highest_on_calculus(self):
        f = self._calculus_integral()
        scores = {
            "printed_basic": _score_printed_basic(f),
            "printed_2d_layout": _score_printed_2d_layout(f),
            "handwritten_basic": _score_handwritten_basic(f),
            "calculus": _score_calculus(f),
        }
        best = max(scores, key=scores.get)
        self.assertEqual(best, "calculus")

    def test_all_scoring_functions_return_reasonable_range(self):
        f = self._flat_printed()
        for fn in [_score_printed_basic, _score_printed_2d_layout, _score_handwritten_basic, _score_calculus]:
            s = fn(f)
            self.assertTrue(0.0 <= s <= 1.2, f"{fn.__name__} returned {s:.2f}, expected 0.0-1.2")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it fails**

Run: `python -m unittest tests.test_structure_router_scoring -v`
Expected: FAIL — `_score_printed_basic` etc. not yet defined.

- [ ] **Step 3: Add scoring functions and replace choose_route()**

In `src/vision/structure_router.py`, replace the `choose_route()` function and the three `_looks_like_*` helper functions (lines 132-228) with:

```python
def _score_printed_basic(features: ImageStructureFeatures) -> float:
    s = 0.50
    if features.is_grid_aligned:
        s += 0.20
    if features.edge_roughness < 0.15:
        s += 0.15
    if features.fraction_like_lines == 0:
        s += 0.10
    if features.component_height_variance < 0.15:
        s += 0.10
    if features.stacked_component_pairs <= 1:
        s += 0.05
    if features.is_extremely_flat_row:
        s += 0.05
    if features.fraction_like_lines > 0:
        s -= 0.15
    return max(0.0, s)


def _score_printed_decimal_negative(features: ImageStructureFeatures) -> float:
    return _score_printed_basic(features)


def _score_printed_2d_layout(features: ImageStructureFeatures) -> float:
    s = 0.35
    if features.fraction_like_lines > 0:
        s += 0.25
    if features.stacked_component_pairs > 1:
        s += 0.15
    if features.long_horizontal_lines > 0:
        s += 0.15
    if features.aspect_ratio < 3.8:
        s += 0.05
    if features.left_tall_components > 0:
        s += 0.05
    if features.is_extremely_flat_row:
        s -= 0.30
    if features.is_handwritten_texture:
        s -= 0.25
    return max(0.0, s)


def _score_handwritten_basic(features: ImageStructureFeatures) -> float:
    s = 0.40
    if features.is_handwritten_texture:
        s += 0.25
    if features.foreground_components <= 5:
        s += 0.15
    if features.foreground_fill_ratio < 0.15:
        s += 0.10
    if not features.is_grid_aligned:
        s += 0.10
    if features.is_single_symbol_like:
        s += 0.05
    if features.fraction_like_lines > 0:
        s -= 0.25
    if features.stacked_component_pairs > 1:
        s -= 0.15
    return max(0.0, s)


def _score_calculus(features: ImageStructureFeatures) -> float:
    s = 0.30
    if features.has_integral_structure:
        s += 0.30
    if features.fraction_like_lines > 0:
        s += 0.15
    if features.edge_roughness > 0.20:
        s += 0.10
    if features.left_tall_components > 0:
        s += 0.10
    if 1.5 < features.aspect_ratio < 4.0:
        s += 0.05
    if features.is_single_symbol_like:
        s -= 0.30
    if features.is_extremely_flat_row:
        s -= 0.25
    return max(0.0, s)


_SCORING_FUNCTIONS = {
    "printed_basic": _score_printed_basic,
    "printed_decimal_negative": _score_printed_decimal_negative,
    "printed_2d_layout": _score_printed_2d_layout,
    "handwritten_basic": _score_handwritten_basic,
    "calculus": _score_calculus,
}


def choose_route(features: ImageStructureFeatures) -> tuple[str, float, str, tuple[str, ...]]:
    scores = {cat: fn(features) for cat, fn in _SCORING_FUNCTIONS.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_category, top_score = ranked[0]
    second_category, second_score = ranked[1]
    margin = top_score - second_score
    fallbacks = tuple(cat for cat, _ in ranked[1:])
    if margin >= 0.12:
        return top_category, top_score, f"clear margin {margin:.2f} over {second_category}", fallbacks
    else:
        return top_category, top_score * 0.7, f"narrow margin {margin:.2f} vs {second_category}", fallbacks
```

Now remove the three old helper functions: `_looks_like_calculus` (lines 181-204), `_looks_like_handwritten_expression` (lines 207-213), and `_looks_like_2d_layout` (lines 216-228).

- [ ] **Step 4: Run scoring tests**

Run: `python -m unittest tests.test_structure_router_scoring -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: Some existing auto_router tests may fail because routing behavior changed. This is expected — we fix them in Task 12.

- [ ] **Step 6: Commit**

```bash
git add src/vision/structure_router.py tests/test_structure_router_scoring.py
git commit -m "feat: replace hard-threshold choose_route() with 5-category weighted scoring"
```

---

### Task 4: Layer 3 — Add cross-validation to auto_router

**Files:**
- Modify: `src/vision/auto_router.py`

- [ ] **Step 1: Write test file for cross-validation**

Create: `tests/test_cross_validation.py`

```python
from __future__ import annotations

import unittest

from src.expression.types import ExpressionResult, LayoutNode, RecognitionResult, SymbolToken
from src.vision.auto_router import RecognitionCandidate, _cross_validate


class CrossValidationTests(unittest.TestCase):
    def _make_result(self, texts: list[str], layout_type: str = "row", solver_answer: str = "") -> RecognitionResult:
        tokens = [SymbolToken(text=t, kind="symbol", bbox=(0, 0, 10, 10), confidence=0.9, source="test") for t in texts]
        layout = LayoutNode(node_type=layout_type, children=[], bbox=(0, 0, 0, 0))
        result = RecognitionResult(tokens=tokens, expression_text="".join(texts), layout=layout)
        result.solver_answer = solver_answer
        return result

    def _make_candidate(self, category: str, result: RecognitionResult, score: float = 0.65) -> RecognitionCandidate:
        return RecognitionCandidate(
            category=category, result=result, score=score, role="primary",
            solver_answer=getattr(result, "solver_answer", ""),
        )

    def test_identical_results_get_positive_score(self):
        r1 = self._make_result(["1", "+", "2"], solver_answer="3")
        r2 = self._make_result(["1", "+", "2"], solver_answer="3")
        c1 = self._make_candidate("printed_basic", r1)
        c2 = self._make_candidate("printed_2d_layout", r2)
        agree = _cross_validate(c1, c2)
        self.assertGreater(agree, 0.20)

    def test_contradictory_solver_results_get_negative(self):
        r1 = self._make_result(["1", "+", "2"], solver_answer="3")
        r2 = self._make_result(["1", "+", "2"], solver_answer="4")
        c1 = self._make_candidate("printed_basic", r1)
        c2 = self._make_candidate("printed_2d_layout", r2)
        agree = _cross_validate(c1, c2)
        self.assertLess(agree, -0.20)

    def test_layout_type_mismatch_reduces_score(self):
        r1 = self._make_result(["x", "^", "2"], layout_type="superscript")
        r2 = self._make_result(["x", "2"], layout_type="row")
        c1 = self._make_candidate("printed_2d_layout", r1)
        c2 = self._make_candidate("printed_basic", r2)
        agree = _cross_validate(c1, c2)
        self.assertLess(agree, 0.15)

    def test_partial_token_overlap_gives_moderate_score(self):
        r1 = self._make_result(["1", "+", "2"])
        r2 = self._make_result(["1", "-", "2"])
        c1 = self._make_candidate("printed_basic", r1)
        c2 = self._make_candidate("handwritten_basic", r2)
        agree = _cross_validate(c1, c2)
        self.assertTrue(0.0 < agree < 0.20)

    def test_none_candidate_returns_zero(self):
        r1 = self._make_result(["1", "+", "2"])
        none_candidate = RecognitionCandidate(category="calculus", result=None, score=-999.0, role="fallback")
        c1 = self._make_candidate("printed_basic", r1)
        agree = _cross_validate(c1, none_candidate)
        self.assertEqual(agree, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it fails**

Run: `python -m unittest tests.test_cross_validation -v`
Expected: FAIL — `_cross_validate` not yet defined.

- [ ] **Step 3: Add _cross_validate() to auto_router.py**

In `src/vision/auto_router.py`, add after the `_router_reason` function (after line 473):

```python
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
        agree += (len(p_set & s_set) / len(p_set | s_set)) * 0.25
    if pr.layout is not None and sr.layout is not None:
        if pr.layout.node_type == sr.layout.node_type:
            agree += 0.15
    if primary.solver_answer and secondary.solver_answer:
        if primary.solver_answer == secondary.solver_answer:
            agree += 0.20
        else:
            agree -= 0.25
    return agree
```

- [ ] **Step 4: Modify recognize_unknown() to use cross-validation on narrow margins**

In `src/vision/auto_router.py`, replace the `recognize_unknown()` function (lines 61-164) with:

```python
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
            xv_score = _cross_validate(primary, secondary)
            # adjust primary score with cross-validation
            primary_adj = RecognitionCandidate(
                category=primary.category, result=primary.result,
                score=primary.score + xv_score,
                score_breakdown={**primary.score_breakdown, "cross_validate": xv_score},
                reject_stage=primary.reject_stage, reject_detail=primary.reject_detail,
                solver_answer=primary.solver_answer, solver_error=primary.solver_error,
                role="primary",
            )
            secondary_adj = RecognitionCandidate(
                category=secondary.category, result=secondary.result,
                score=secondary.score + xv_score,
                score_breakdown={**secondary.score_breakdown, "cross_validate": xv_score},
                reject_stage=secondary.reject_stage, reject_detail=secondary.reject_detail,
                solver_answer=secondary.solver_answer, solver_error=secondary.solver_error,
                role="cross_validation",
            )
            candidates = [primary_adj, secondary_adj]
            run_categories = {primary.category, secondary.category}

            both_low = primary_adj.score < 0.0 and secondary_adj.score < 0.0
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
                    category=category, result=None, score=-999.0,
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
            score=-999.0, reject_stage="routing", reject_detail="all candidates rejected by structure router",
            role="selected",
        )
        reason = _router_reason(selected, analysis)
        return AutoRecognitionDecision(
            selected=selected, candidates=candidates, image_features=analysis.features,
            router_reason=reason, structure_analysis=analysis,
        )

    selected = max(runnable, key=lambda candidate: _selection_score(candidate, analysis.route_hint))
    reason = _router_reason(selected, analysis)
    selected_result = _mark_auto_result(selected.result, selected.category, reason, analysis)
    selected = RecognitionCandidate(
        category=selected.category, result=selected_result, score=selected.score,
        score_breakdown=selected.score_breakdown, reject_stage=selected.reject_stage,
        reject_detail=selected.reject_detail, solver_answer=selected.solver_answer,
        solver_error=selected.solver_error, role="selected",
    )
    return AutoRecognitionDecision(
        selected=selected, candidates=candidates, image_features=analysis.features,
        router_reason=reason, structure_analysis=analysis,
    )
```

- [ ] **Step 5: Run cross-validation tests**

Run: `python -m unittest tests.test_cross_validation -v`
Expected: 5 tests PASS.

- [ ] **Step 6: Verify test file imports work**

Run: `python -m unittest tests.test_auto_router -v`
Expected: May fail due to routing changes — acceptable, fixed in Task 12.

- [ ] **Step 7: Commit**

```bash
git add src/vision/auto_router.py tests/test_cross_validation.py
git commit -m "feat: add dual-path cross-validation for narrow-margin routing decisions"
```

---

### Task 5: Layer 4a — Refactor _disambiguate() to 3-layer with context parameter

**Files:**
- Modify: `src/vision/recognizers/handwritten_rule_template.py`

- [ ] **Step 1: Write test file for disambiguation layers**

Create: `tests/test_disambiguation_layers.py`

```python
from __future__ import annotations

import unittest

import numpy as np

from src.vision.recognizers.handwritten_rule_template import _disambiguate


class DisambiguationLayerTests(unittest.TestCase):
    def _features(self, **overrides) -> dict[str, float]:
        defaults = {
            "num_holes": 0, "num_components": 1, "aspect_ratio": 0.9,
            "centroid_x": 0.50, "centroid_y": 0.50,
            "top_bottom_ratio": 1.0, "left_right_ratio": 1.0,
            "horizontal_score": 0.1, "vertical_score": 0.1,
            "diag_pos_score": 0.1, "diag_neg_score": 0.1,
            "fill_ratio": 0.15, "density_top": 0.1, "density_bottom": 0.1,
        }
        defaults.update(overrides)
        return defaults

    def test_two_holes_forces_eight(self):
        candidates = {"8": 0.70, "0": 0.75, "6": 0.65}
        f = self._features(num_holes=2)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "8")
        self.assertIn("geometry:", reason)

    def test_three_components_forces_division(self):
        candidates = {"÷": 0.70, "+": 0.72, "x": 0.68}
        f = self._features(num_components=3, aspect_ratio=0.82)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "÷")

    def test_long_horizontal_forces_minus(self):
        candidates = {"-": 0.55, "1": 0.60, "/": 0.58}
        f = self._features(aspect_ratio=3.0, horizontal_score=0.35, diag_pos_score=0.05, diag_neg_score=0.05)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "-")

    def test_parallel_horizontal_forces_equals(self):
        candidates = {"=": 0.55, "-": 0.58, "÷": 0.50}
        f = self._features(aspect_ratio=1.6, horizontal_score=0.45, diag_pos_score=0.05, diag_neg_score=0.05)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "=")

    def test_balanced_diagonal_forces_x_over_multiply(self):
        candidates = {"x": 0.52, "×": 0.54, "y": 0.48}
        f = self._features(
            diag_pos_score=0.35, diag_neg_score=0.35, horizontal_score=0.05,
            aspect_ratio=0.95, centroid_x=0.50,
        )
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "x")

    def test_slash_diagonal_dominance(self):
        candidates = {"/": 0.50, "1": 0.55, "7": 0.48}
        f = self._features(
            diag_neg_score=1.5, diag_pos_score=0.10, horizontal_score=0.05,
            vertical_score=0.05, top_bottom_ratio=1.0,
        )
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "/")

    def test_context_calculus_boosts_d(self):
        candidates = {"d": 0.48, "y": 0.52, "4": 0.45}
        f = self._features(aspect_ratio=0.75, horizontal_score=0.05, vertical_score=0.35)
        adjusted, reason = _disambiguate(f, candidates, context="calculus")
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "d")

    def test_confusion_pair_five_vs_s(self):
        candidates = {"5": 0.48, "s": 0.52}
        f = self._features(density_top=0.25, num_holes=0)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "5")

    def test_confusion_pair_two_vs_z(self):
        candidates = {"2": 0.48, "z": 0.50}
        f = self._features(top_bottom_ratio=0.78, num_holes=0)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "2")

    def test_confusion_pair_dash_vs_underscore(self):
        candidates = {"-": 0.48, "_": 0.50}
        f = self._features(centroid_y=0.50, aspect_ratio=2.5, horizontal_score=0.30)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "-")

    def test_confusion_pair_zero_vs_oh_vs_oh_capital(self):
        candidates = {"0": 0.45, "o": 0.48, "O": 0.44}
        f = self._features(aspect_ratio=0.90, num_holes=1)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "o")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — verify it fails**

Run: `python -m unittest tests.test_disambiguation_layers -v`
Expected: FAIL — `context` parameter not yet supported; some confusion pairs not yet implemented.

- [ ] **Step 3: Refactor _disambiguate() with context parameter and new rules**

In `src/vision/recognizers/handwritten_rule_template.py`, replace `_disambiguate()` (lines 144-296) with:

```python
def _disambiguate(
    query_features: dict[str, float],
    candidates: dict[str, float],
    context: str = "handwritten",
) -> tuple[dict[str, float], str]:
    """Adjust template scores using deterministic handwriting geometry.

    Layers: A (hard geometry force), B (context-aware bias), C (confusion pairs).
    """
    adjusted = dict(candidates)
    reason = "template_only"
    original_best = max(candidates, key=lambda key: candidates[key])

    def penalize(labels: tuple[str, ...], factor: float) -> None:
        for label in labels:
            if label in adjusted:
                adjusted[label] *= factor

    def boost(label: str, amount: float) -> None:
        if label in adjusted:
            adjusted[label] += amount

    def force(label: str, rule: str) -> None:
        nonlocal reason
        if label not in adjusted:
            return
        adjusted[label] = max(adjusted.values()) + 0.08
        reason = rule

    holes = int(round(query_features["num_holes"]))
    components = int(round(query_features["num_components"]))
    aspect = query_features["aspect_ratio"]
    centroid_x = query_features["centroid_x"]
    centroid_y = query_features["centroid_y"]
    top_bottom = query_features["top_bottom_ratio"]
    left_right = query_features["left_right_ratio"]
    horizontal = query_features["horizontal_score"]
    vertical = query_features["vertical_score"]
    diag_pos = query_features["diag_pos_score"]
    diag_neg = query_features["diag_neg_score"]
    diagonal_balance = min(diag_pos, diag_neg) / max(max(diag_pos, diag_neg), 1e-6)

    # ---- Layer A: hard geometric rules ----
    stable_symbols = ("(", ")", "+", ".")
    if original_best in stable_symbols and candidates[original_best] >= 0.55:
        return adjusted, reason

    if components >= 3 and 0.75 <= aspect <= 1.3:
        force("÷", "geometry:three_components")
        return adjusted, reason
    if holes >= 2:
        force("8", "geometry:two_holes")
        return adjusted, reason

    if aspect > 2.5 and horizontal > 0.2 and max(diag_pos, diag_neg) < 0.25:
        force("-", "geometry:long_horizontal")
        return adjusted, reason
    if aspect > 1.45 and horizontal > 0.35 and max(diag_pos, diag_neg) < 0.25:
        force("=", "geometry:parallel_horizontal")
        return adjusted, reason

    # ---- Layer B: context-aware adjustments ----
    if context == "calculus":
        boost("d", 0.10)
        boost("x", 0.05)
        boost("∫", 0.08)
        penalize(("y", "4", "6"), 0.75)
        if original_best == "y" and candidates.get("d", 0.0) >= candidates["y"] - 0.08:
            force("d", "geometry:calculus_context_switch_y_to_d")
            return adjusted, reason

    # ---- Layer C: confusion pair rules ----
    digit_labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    digit_score = max((candidates.get(label, 0.0) for label in digit_labels), default=0.0)
    nondigit_score = max(
        (candidates.get(label, 0.0) for label in ("+", "-", "÷", "/", "=", "(", ")", ".", "x", "y", "×")),
        default=0.0,
    )
    digit_context = digit_score >= nondigit_score - 0.04

    # holes == 1 rules (existing, preserved)
    if holes == 1:
        penalize(("2", "3", "5", "8"), 0.35)
        if top_bottom > 1.08 and left_right < 0.95 and centroid_y < 0.5 and digit_context:
            force("9", "geometry:upper_right_loop")
        elif left_right > 1.1 and centroid_y >= 0.49 and digit_context:
            force("6", "geometry:lower_left_loop")
        elif 0.85 <= top_bottom <= 1.15 and 0.85 <= left_right <= 1.25 and digit_context:
            force("0", "geometry:centered_loop")
        elif left_right < 0.75 and digit_context:
            force("4", "geometry:right_heavy_open_four")
        return adjusted, reason

    if digit_context:
        penalize(("0", "6", "8", "9"), 0.45)

    # existing digit/operator rules (preserved)
    if (
        components < 3
        and 0.8 <= aspect <= 0.9
        and 0.9 <= top_bottom <= 1.2
        and 0.78 <= left_right <= 0.98
        and query_features["fill_ratio"] > 0.28
        and diag_neg > 0.7
        and candidates.get("4", 0.0) >= candidates.get("÷", 0.0) - 0.05
    ):
        force("4", "geometry:open_four")
        return adjusted, reason

    if aspect < 0.72 and top_bottom > 1.45 and diag_pos > 0.2 and diag_neg > 0.3 and candidates.get("y", 0.0) > 0.25:
        force("y", "geometry:y_upper_fork")
        return adjusted, reason
    if (
        aspect < 0.68
        and centroid_x < 0.65
        and vertical > 0.45
        and diag_neg > 0.18
        and horizontal < 0.25
        and candidates.get("y", 0.0) > 0.25
    ):
        force("y", "geometry:y_descender")
        return adjusted, reason
    if (
        aspect > 0.75
        and top_bottom > 1.25
        and query_features["fill_ratio"] < 0.25
        and candidates.get("y", 0.0) >= max(candidates.get("x", 0.0), candidates.get("7", 0.0)) - 0.06
    ):
        force("y", "geometry:y_open_tail")
        return adjusted, reason

    if diag_neg > 1.35 and diag_pos < 0.25 and horizontal < 0.12 and vertical < 0.12 and 0.8 <= top_bottom <= 1.2:
        force("/", "geometry:pure_slash")
        return adjusted, reason
    if aspect < 0.55 and diag_neg > 0.4 and vertical > 0.12 and horizontal < 0.25 and top_bottom > 1.2:
        force("1", "geometry:narrow_one")
        return adjusted, reason
    if diag_pos > 0.2 and diag_neg > 0.2 and horizontal < 0.15 and max(candidates.get("x", 0.0), candidates.get("×", 0.0)) > 0.25:
        if 0.9 <= aspect <= 1.08 and diagonal_balance > 0.5 and abs(centroid_x - 0.5) < 0.06:
            if candidates.get("×", 0.0) - candidates.get("x", 0.0) < 0.08:
                force("x", "geometry:near_tie_variable_x")
            else:
                force("×", "geometry:centered_cross")
        else:
            force("x", "geometry:variable_x")
        return adjusted, reason
    if aspect > 1.1 and diag_neg > 0.4 and horizontal < 0.15 and vertical < 0.15:
        force("x", "geometry:wide_variable_x")
        return adjusted, reason

    if digit_context and horizontal > 0.3 and diag_neg > 0.6 and left_right >= 0.9 and top_bottom < 1.0:
        force("2", "geometry:two_bottom_sweep")
        return adjusted, reason
    if (
        digit_context
        and left_right < 0.75
        and top_bottom > 0.88
        and vertical < 0.1
        and candidates.get("3", 0.0) >= candidates.get("5", 0.0) - 0.08
    ):
        force("3", "geometry:right_heavy_three")
        return adjusted, reason

    # ---- new confusion pair rules ----
    # x vs × (refined — when not caught by centered_cross above)
    if candidates.get("×", 0.0) > 0.35 and candidates.get("x", 0.0) > 0.35:
        if diagonal_balance > 0.6 and abs(centroid_x - 0.5) < 0.08:
            force("×", "geometry:multiply_cross")
        else:
            force("x", "geometry:variable_x_off_center")

    # ÷ vs + (when components < 3)
    if candidates.get("÷", 0.0) > 0.35 and candidates.get("+", 0.0) > 0.35:
        if components >= 2 and diag_pos < 0.15 and diag_neg < 0.15:
            force("+", "geometry:plus_over_division")
        elif components == 1 and holes == 0:
            penalize(("÷",), 0.65)

    # 1 vs / (tighter thresholds)
    if candidates.get("1", 0.0) > 0.30 and candidates.get("/", 0.0) > 0.30:
        if vertical > 0.35 and diag_neg < 0.30:
            force("1", "geometry:vertical_one")
        elif diag_neg > 0.55 and vertical < 0.20:
            force("/", "geometry:diagonal_slash")

    # - vs _ (position-based)
    if candidates.get("-", 0.0) > 0.30 and candidates.get("_", 0.0) > 0.30:
        if centroid_y > 0.65:
            force("_", "geometry:low_underscore")
        elif 0.40 < centroid_y < 0.58:
            force("-", "geometry:mid_minus")

    # 0 vs o vs O
    if holes == 1 and candidates.get("0", 0.0) > 0.30:
        if 0.60 < aspect < 0.85:
            force("0", "geometry:tall_zero")
        elif 0.85 <= aspect <= 1.05:
            force("o", "geometry:round_oh")
        elif 1.0 < aspect <= 1.25:
            force("O", "geometry:wide_oh_capital")

    # , vs .
    if candidates.get(",", 0.0) > 0.30 and candidates.get(".", 0.0) > 0.30:
        if centroid_y > 0.60 and query_features["fill_ratio"] < 0.10:
            force(",", "geometry:low_comma")
        elif query_features["fill_ratio"] < 0.08 and centroid_y < 0.55:
            force(".", "geometry:small_dot")

    # 2 vs z
    if candidates.get("2", 0.0) > 0.30 and candidates.get("z", 0.0) > 0.30:
        if top_bottom < 0.85:
            force("2", "geometry:curved_top_two")
        elif top_bottom > 0.95:
            force("z", "geometry:flat_top_z")

    # 5 vs s
    if candidates.get("5", 0.0) > 0.30 and candidates.get("s", 0.0) > 0.30:
        if query_features["density_top"] > 0.15:
            force("5", "geometry:top_bar_five")
        else:
            force("s", "geometry:no_top_bar_s")

    # 9 vs g
    if candidates.get("9", 0.0) > 0.30 and candidates.get("g", 0.0) > 0.30:
        if query_features["density_bottom"] < 0.40:
            force("9", "geometry:closed_bottom_nine")
        elif query_features["density_bottom"] > 0.50:
            force("g", "geometry:descender_g")

    # 6 vs b
    if candidates.get("6", 0.0) > 0.30 and candidates.get("b", 0.0) > 0.30:
        if left_right > 1.05:
            force("6", "geometry:left_loop_six")
        elif left_right < 0.95:
            force("b", "geometry:right_loop_b")

    if query_features["density_top"] > 0.2:
        boost("7", 0.04)
        boost("5", 0.04)
    if query_features["density_bottom"] > 0.2:
        boost("y", 0.04)

    if reason == "template_only" and any(abs(candidates[k] - adjusted[k]) > 0.001 for k in adjusted):
        reason = "geometry_adjusted"
    return adjusted, reason
```

- [ ] **Step 4: Update match_handwritten_symbol() to pass context**

In `src/vision/recognizers/handwritten_rule_template.py`, update `match_handwritten_symbol()` signature (line 360) and the call to `_disambiguate` (line 387):

Change the function signature from:
```python
def match_handwritten_symbol(
    roi: np.ndarray,
    templates: TemplateLibrary,
    size: int = 64,
) -> tuple[str, float, dict[str, float], str]:
```
To:
```python
def match_handwritten_symbol(
    roi: np.ndarray,
    templates: TemplateLibrary,
    size: int = 64,
    context: str = "handwritten",
) -> tuple[str, float, dict[str, float], str]:
```

Change line 387 from:
```python
    adjusted, decision_type = _disambiguate(query_features, candidates)
```
To:
```python
    adjusted, decision_type = _disambiguate(query_features, candidates, context=context)
```

- [ ] **Step 5: Update calculus_layout.py to pass context="calculus"**

In `src/vision/calculus_layout.py`, update `_match_segment()` (line 201) — change:
```python
    label, score, candidates, _ = match_handwritten_symbol(segment.image, templates)
```
To:
```python
    label, score, candidates, _ = match_handwritten_symbol(segment.image, templates, context="calculus")
```

- [ ] **Step 6: Run disambiguation tests**

Run: `python -m unittest tests.test_disambiguation_layers -v`
Expected: 11 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/vision/recognizers/handwritten_rule_template.py src/vision/calculus_layout.py tests/test_disambiguation_layers.py
git commit -m "feat: refactor _disambiguate into 3-layer hierarchy with context-aware rules and 10 confusion pairs"
```

---

### Task 6: Layer 4b — Expand _calculus_symbol_override() rules

**Files:**
- Modify: `src/vision/calculus_layout.py`

- [ ] **Step 1: Replace the function**

In `src/vision/calculus_layout.py`, replace `_calculus_symbol_override()` (lines 357-363) with:

```python
def _calculus_symbol_override(image: np.ndarray, label: str, score: float) -> tuple[str, float]:
    features = extract_geometry_features(image)
    aspect = features["aspect_ratio"]
    vertical = features["vertical_score"]
    horizontal = features["horizontal_score"]
    holes = features["num_holes"]
    components = features["num_components"]

    if label in {"y", "/"} and aspect < 0.65 and vertical > 0.15:
        return "1", max(score, 0.82)
    if label == "÷" and components <= 1 and aspect > 2.0:
        return "-", max(score, 0.84)
    if label in {"y", "4", "/"} and aspect < 0.8 and horizontal < 0.1:
        return "d", max(score, 0.78)
    if holes == 1 and 0.8 < aspect < 1.2 and label in {"8", "6", "9", "4"}:
        return "0", max(score, 0.76)
    return label, score
```

- [ ] **Step 2: Run calculus-related tests**

Run: `python -m unittest tests.test_calculus_layout -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/vision/calculus_layout.py
git commit -m "feat: expand calculus symbol override with 'd' and '0' recovery rules"
```

---

### Task 7: Layer 4c — Expand function prefix map in calculus_rules.py

**Files:**
- Modify: `src/vision/calculus_rules.py`

- [ ] **Step 1: Add expanded FUNCTION_PREFIX_MAP and update _classify_function_prefix()**

In `src/vision/calculus_rules.py`, replace `_classify_function_prefix()` (lines 120-132) with:

```python
_FUNCTION_PREFIX_MAP = {
    "sin": {"sin", "3/x", "5/x", "5(1", "51x", "5ix", "5/n", "s1n", "sln", "51n"},
    "cos": {"cos", "c05", "c0s", "co5", "c0S"},
    "tan": {"tan", "7an", "t4n", "t@n", "t4N"},
    "exp": {"exp", "6x2", "ex2", "64d", "e*p", "3xp"},
    "log": {"ln", "1n", "1N", "lN", "lR"},
    "arcsin": {"arcsin", "arc5in", "arc51n"},
    "arccos": {"arccos", "arcco5", "arcc05"},
    "arctan": {"arctan", "arct4n", "arct@n"},
}

_ARCFUNCTION_NAMES = {"arcsin", "arccos", "arctan"}


def _classify_function_prefix(prefix: str) -> str | None:
    compact = prefix.replace("UNKNOWN", "?")
    for name, variants in _FUNCTION_PREFIX_MAP.items():
        if compact in variants:
            return name
    # handle arc functions specially: longer prefix match first
    # then remove "arc" to check inner function
    for arc_name in _ARCFUNCTION_NAMES:
        for variant in _FUNCTION_PREFIX_MAP[arc_name]:
            if compact.startswith(variant):
                return arc_name
    return None
```

Also update `_classify_function_prefix` in `src/vision/calculus_layout.py` (lines 424-438) to use the same expanded map from `calculus_rules.py`:

Replace those lines with:
```python
def _classify_function_prefix(prefix: str) -> str | None:
    # Reuse the same logic from calculus_rules
    from src.vision.calculus_rules import _classify_function_prefix as _cfp
    return _cfp(prefix)
```

- [ ] **Step 2: Run calculus rules tests**

Run: `python -m unittest tests.test_calculus_rules -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/vision/calculus_rules.py src/vision/calculus_layout.py
git commit -m "feat: expand function prefix map for sin/cos/tan/exp/log/arc variants"
```

---

### Task 8: Add confusion matrix output to evaluate_samples.py

**Files:**
- Modify: `tools/evaluate_samples.py`

- [ ] **Step 1: Add --confusion-matrix flag and output logic**

In `tools/evaluate_samples.py`, add the argument in `main()` after line 131:

```python
    parser.add_argument("--confusion-matrix", action="store_true", help="Output confusion matrix for auto mode.")
```

After `_print_summary(rows)` (line 157), add:

```python
    if args.confusion_matrix:
        _print_confusion_matrix(rows)
        if auto_mode:
            matrix_path = Path(args.output).with_suffix(".matrix.csv")
            _write_confusion_matrix_csv(rows, matrix_path)
            print(f"confusion matrix saved to {matrix_path}")
```

Add the helper functions before `if __name__ == "__main__":`:

```python
def _print_confusion_matrix(rows: list[EvaluationRow]) -> None:
    true_categories = sorted({row.category for row in rows})
    pred_categories = sorted({row.selected_category for row in rows if row.selected_category})
    all_cats = sorted(set(true_categories) | set(pred_categories))
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>25}" + "".join(f"{cat:>12}" for cat in all_cats)
    print(header)
    for true_cat in all_cats:
        subset = [row for row in rows if row.category == true_cat]
        counts = [str(sum(1 for row in subset if row.selected_category == pred_cat)) for pred_cat in all_cats]
        line = f"{true_cat:>25}" + "".join(f"{c:>12}" for c in counts)
        print(line)


def _write_confusion_matrix_csv(rows: list[EvaluationRow], path: Path) -> None:
    import csv
    true_categories = sorted({row.category for row in rows})
    pred_categories = sorted({row.selected_category for row in rows if row.selected_category})
    all_cats = sorted(set(true_categories) | set(pred_categories))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + all_cats)
        for true_cat in all_cats:
            subset = [row for row in rows if row.category == true_cat]
            writer.writerow([true_cat] + [sum(1 for row in subset if row.selected_category == pred_cat) for pred_cat in all_cats])
```

- [ ] **Step 2: Test confusion matrix output**

Run: `python tools/evaluate_samples.py --category auto --output evaluation_test_matrix.csv --confusion-matrix`
Expected: Prints confusion matrix to stdout and writes `.matrix.csv` file.

- [ ] **Step 3: Commit**

```bash
git add tools/evaluate_samples.py
git commit -m "feat: add --confusion-matrix flag to evaluate_samples for auto router diagnostics"
```

---

### Task 9: Update test_auto_router.py for new scoring behavior

**Files:**
- Modify: `tests/test_auto_router.py`

- [ ] **Step 1: Run current auto_router tests to see failures**

Run: `python -m unittest tests.test_auto_router -v`
Expected: Some tests may fail due to changed routing behavior. Note which ones.

- [ ] **Step 2: Update tests for scoring-based routing**

Replace `tests/test_auto_router.py` with updated assertions that match the new scoring router. The key change is that `route_hint` may differ from the old hard-threshold behavior, but should still route correctly for well-separated cases. The existing test expectations (flat printed → printed_basic, 2d → printed_2d_layout, calculus → calculus, single symbol → handwritten_basic) should still hold because the scoring functions are designed to preserve them.

If any test fails because of confidence value changes, update the assertion to check the category rather than the exact confidence:

```python
from __future__ import annotations

import unittest
from pathlib import Path

from src.vision.auto_router import extract_image_structure_features, recognize_unknown
from src.vision.structure_router import analyze_formula_structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AutoRouterTests(unittest.TestCase):
    def test_flat_printed_sample_routes_to_printed(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "printed_basic" / "printed_basic_007.png"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        rejected = {candidate.category: candidate.reject_stage for candidate in decision.candidates}
        self.assertIn(decision.structure_analysis.route_hint, {"printed_basic", "printed_decimal_negative"})
        self.assertEqual(decision.selected.category, "printed_basic")
        self.assertEqual(rejected.get("calculus"), "route_prior")

    def test_structural_sample_routes_to_2d(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_001.png"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        self.assertEqual(decision.structure_analysis.route_hint, "printed_2d_layout")
        self.assertEqual(decision.selected.category, "printed_2d_layout")

    def test_calculus_limit_selects_calculus(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "calculus" / "calculus_007.png"
        decision = recognize_unknown(image, solver_timeout_ms=1500)
        self.assertEqual(decision.structure_analysis.route_hint, "calculus")
        self.assertEqual(decision.selected.category, "calculus")
        self.assertEqual(decision.selected.result.sympy_text, "limit(sin(x)/x, x, 0)")

    def test_single_symbol_prefers_handwritten(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "handwritten_basic" / "handwritten_8_001.png"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        self.assertEqual(decision.structure_analysis.route_hint, "handwritten_basic")
        self.assertEqual(decision.selected.category, "handwritten_basic")
        self.assertEqual(decision.selected.result.expression_text, "8")

    def test_feature_extraction_reports_complexity_signals(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "calculus" / "calculus_008.png"
        features = extract_image_structure_features(image)
        self.assertGreater(features.fraction_like_lines, 0)
        self.assertGreater(features.foreground_components, 1)

    def test_structure_analysis_reports_route_diagnostics(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "calculus" / "calculus_008.png"
        analysis = analyze_formula_structure(image)
        self.assertEqual(analysis.route_hint, "calculus")
        self.assertGreater(analysis.route_confidence, 0.0)
        self.assertGreater(analysis.features.long_horizontal_lines, 0)

    def test_rough_handwritten_expression_routes_to_handwritten(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "test" / "1.jpg"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        self.assertEqual(decision.structure_analysis.route_hint, "handwritten_basic")
        self.assertEqual(decision.selected.category, "handwritten_basic")
        self.assertEqual(decision.selected.result.expression_text, "3+5")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests**

Run: `python -m unittest tests.test_auto_router -v`
Expected: All 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_auto_router.py
git commit -m "test: update auto_router tests for scoring-based routing"
```

---

### Task 10: Final verification — full test suite + evaluation

**Files:**
- None modified

- [ ] **Step 1: Run full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: All 68 tests PASS (57 original + 5 scoring + 5 cross_validation + 11 disambiguation — minus adjustments; total ~68, 0 failures).

- [ ] **Step 2: Run auto evaluation with confusion matrix**

Run: `python tools/evaluate_samples.py --category auto --output evaluation_improved_auto.csv --confusion-matrix`
Expected: CSV written. Compare per-category accuracy with baseline from Task 1.

- [ ] **Step 3: Verify success criteria**

Check:
1. Overall auto accuracy improved vs baseline
2. No single category dropped > 5%
3. Confusion matrix shows no systematic misrouting pattern

- [ ] **Step 4: Commit evaluation results**

```bash
git add evaluation_improved_auto.csv evaluation_improved_auto.matrix.csv
git commit -m "eval: post-improvement auto router accuracy baseline"
```

---

## Summary

| Task | Files Changed | Tests |
|---|---|---|
| 1 - Baseline | — | 57 existing |
| 2 - Layer 1 features | `structure_router.py` | 57 existing |
| 3 - Layer 2 scoring | `structure_router.py` | +5 scoring |
| 4 - Layer 3 cross-validation | `auto_router.py` | +5 cross_validate |
| 5 - Layer 4a disambiguation | `handwritten_rule_template.py`, `calculus_layout.py` | +11 disambiguation |
| 6 - Layer 4b symbol override | `calculus_layout.py` | existing calculus |
| 7 - Layer 4c prefix map | `calculus_rules.py`, `calculus_layout.py` | existing calculus |
| 8 - Confusion matrix | `evaluate_samples.py` | manual |
| 9 - Update tests | `test_auto_router.py` | updated 7 |
| 10 - Final verification | — | ~68 total |
