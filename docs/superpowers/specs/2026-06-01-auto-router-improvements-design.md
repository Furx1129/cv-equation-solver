# Auto Router & Disambiguation Improvements — Design Spec

**Date:** 2026-06-01
**Status:** approved, awaiting implementation plan
**Scope:** 改进 auto 路由准确率 + 易混淆符号消歧

---

## Goals

1. 将 `choose_route()` 从硬阈值 if-else 链改为 5 类别同时打分制
2. 对路由边距不足的样本增加双路并行交叉验证
3. 将 `_disambiguate()` 重构为三层规则架构（通用 → 类别感知 → 混淆对）
4. 补充 10+ 组已知混淆对的几何区分规则
5. 测量改进效果：整体 auto 准确率 + 每类别混淆矩阵

## Non-Goals

- 不增加新的图片样本
- 不改动求解器（ArithmeticSolver / SymbolicSolver）
- 不改动预处理管线（preprocess / deskew / normalization / segmentation）
- 不引入 ML/DL 方法
- 不改动 printed_template.py 识别器
- 不增加 GUI

---

## Layer 1: Enhanced Feature Extraction

**File:** `src/vision/structure_router.py`

### New fields on `ImageStructureFeatures`

| Field | Computation | Purpose |
|---|---|---|
| `vertical_symmetry` | `1 - abs(top_half_fg - bottom_half_fg) / total_fg` | Printed formulas are symmetric about the baseline; handwritten are not |
| `component_height_variance` | `std(component_heights) / mean(component_heights)` | Handwritten has higher height variance than printed |
| `horizontal_run_count` | Number of runs in horizontal projection above noise floor | Distinguishes multi-line vs single-line |
| `left_right_density_ratio` | `fg_left_half / max(1, fg_right_half)` | Integral signs are left-heavy |
| `multi_scale_edge_ratio` | `edges_low_threshold / max(1, edges_high_threshold)` | Printed edges are sharp (ratio ~1); handwritten are fuzzy (ratio >1.5) |
| `top_alignment_score` | `1 / (1 + std(component_top_y_coords))` | Printed symbols align to a common top; handwritten drift |

### New computed properties

```python
is_vertically_symmetric: bool     # vertical_symmetry > 0.7
is_handwritten_texture: bool      # multi_scale_edge_ratio > 1.8 AND height_variance > 0.3
has_integral_structure: bool      # left_right_density_ratio > 1.5 AND left_tall_components > 0
is_grid_aligned: bool             # top_alignment_score > 3.0
```

### Backward compatibility

All existing fields and properties are preserved. `as_dict()` gains the new fields. Tests that check `ImageStructureFeatures` fields are updated to include new defaults.

---

## Layer 2: Full-Category Scoring Router

**File:** `src/vision/structure_router.py` — replaces `choose_route()` body.

### Scoring functions (one per category)

Each scoring function starts from a baseline and adds weighted feature contributions. Initial weights are set from domain knowledge; they will be calibrated after running evaluation.

```
_score_printed_basic(f):
  base 0.50
  + is_grid_aligned?             +0.20
  + edge_roughness < 0.15?       +0.15
  + fraction_like_lines == 0?    +0.10
  + height_variance < 0.15?      +0.10
  + stacked_pairs <= 1?          +0.05
  + is_extremely_flat_row?       +0.05
  - fraction_like_lines > 0?     -0.15

_score_printed_decimal_negative(f):
  # Same recognizer as printed_basic; distinction happens via _display_category_for_result()
  return _score_printed_basic(f)

_score_printed_2d_layout(f):
  base 0.35
  + fraction_like_lines > 0?     +0.25
  + stacked_pairs > 1?           +0.15
  + long_horizontal_lines > 0?   +0.15
  + aspect_ratio < 3.8?          +0.05
  + left_tall_components > 0?    +0.05
  - is_extremely_flat_row?       -0.30
  - is_handwritten_texture?      -0.25

_score_handwritten_basic(f):
  base 0.40
  + is_handwritten_texture?      +0.25
  + components <= 5?             +0.15
  + fill_ratio < 0.15?           +0.10
  + not is_grid_aligned?         +0.10
  + is_single_symbol_like?       +0.05
  - fraction_like_lines > 0?     -0.25
  - stacked_pairs > 1?           -0.15

_score_calculus(f):
  base 0.30
  + has_integral_structure?      +0.30
  + fraction_like_lines > 0?     +0.15
  + edge_roughness > 0.20?       +0.10
  + left_tall_components > 0?    +0.10
  + 1.5 < aspect_ratio < 4.0?    +0.05
  - is_single_symbol_like?       -0.30
  - is_extremely_flat_row?       -0.25
```

### Routing decision

```python
def choose_route(features):
    scores = {cat: fn(features) for cat, fn in SCORING_FUNCTIONS.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, second = ranked[0], ranked[1]
    margin = top[1] - second[1]
    fallbacks = tuple(cat for cat, _ in ranked[1:])
    if margin >= 0.12:
        return top[0], top[1], f"clear margin {margin:.2f}", fallbacks
    else:
        return top[0], top[1] * 0.7, f"narrow margin {margin:.2f}", fallbacks
```

When `margin < 0.12`, the lowered confidence signals `recognize_unknown()` to activate Layer 3 cross-validation.

---

## Layer 3: Dual-Path Cross-Validation

**File:** `src/vision/auto_router.py` — new function `_cross_validate()` and modified flow.

### Trigger

Activated when `FormulaStructureAnalysis.route_confidence <= 0.7` (i.e., margin was narrow). The top-2 categories from Layer 2 are both run through their respective recognizers in parallel via `ThreadPoolExecutor`.

### Cross-validation function

```python
def _cross_validate(primary, secondary):
    agree = 0.0
    # Token overlap
    if primary and secondary:
        p_set = {t.text for t in primary.tokens if t.text != "UNKNOWN"}
        s_set = {t.text for t in secondary.tokens if t.text != "UNKNOWN"}
        if p_set and s_set:
            agree += (len(p_set & s_set) / len(p_set | s_set)) * 0.25
    # Layout type match
    if primary.layout and secondary.layout:
        if primary.layout.node_type == secondary.layout.node_type:
            agree += 0.15
    # Solver agreement
    if primary.solver_answer and secondary.solver_answer:
        agree += 0.20 if primary.solver_answer == secondary.solver_answer else -0.25
    return agree
```

### Selection

Each candidate's final score = `_score_candidate(...)` + `_cross_validate(...)`. The candidate with the highest adjusted score is selected. If cross-validation fails to resolve (both scores < 0), continue fallback to the 3rd-ranked category.

---

## Layer 4: Hierarchical Symbol Disambiguation

### 4a. Refactored `_disambiguate()` in `handwritten_rule_template.py`

**New signature:**

```python
def _disambiguate(
    query_features: dict[str, float],
    candidates: dict[str, float],
    context: str = "handwritten",
) -> tuple[dict[str, float], str]:
```

**Layer A — hard geometric rules (unchanged logic, extracted to standalone helpers):**
- `holes >= 2 → force("8")`
- `components >= 3 and 0.75 <= aspect <= 1.3 → force("÷")`
- `aspect > 2.5 and horizontal strong → force("-")`
- etc.

**Layer B — context-aware adjustments (new):**

| Context | Bias |
|---|---|
| `"handwritten"` | No special bias; keep existing digit/operator discrimination |
| `"calculus"` | Boost `d`, `∫`, `x`; penalize digits that look like calculus symbols |
| `"auto"` | Delegates to `"handwritten"` for isolated symbols, `"calculus"` when `has_integral_structure` is true |

**Layer C — confusion pair rules (existing + new):**

| Confusion Pair | Rule |
|---|---|
| `x` vs `×` | `×` has balanced diagonal symmetry; `x` has off-center crossing. Use `diagonal_balance` and `centroid_x` proximity to 0.5 |
| `÷` vs `+` | `÷` has 3 components; `+` has 2 lines that cross. Use `num_components` and aspect constraints |
| `1` vs `/` | `1` is vertical dominant (vertical_score > 0.4); `/` is diagonal dominant (diag_neg > 0.5). Already partially covered; add tighter thresholds |
| `-` vs `_` | `-` sits near vertical center (centroid_y ~0.45-0.55); `_` sits low (centroid_y > 0.7) |
| `0` vs `o` vs `O` | `0` is tall (aspect 0.6-0.85); `o` is round (0.85-1.05); `O` is wide (1.0-1.2) |
| `,` vs `.` | `,` is below baseline with tail (centroid_y > 0.65); `.` is small dot near baseline |
| `2` vs `z` | `2` has curved top (top_bottom_ratio < 0.85); `z` has flat top (top_bottom_ratio ~1.0) |
| `5` vs `s` | `5` has top horizontal bar (density_top > 0.15); `s` has no bar |
| `9` vs `g` | `9` bottom is closed (density_bottom < 0.4); `g` has descender tail (density_bottom > 0.5) |
| `6` vs `b` | `6` has left-side loop (left_right_ratio > 1.05); `b` has right-side loop (left_right_ratio < 0.95) |

### 4b. Calculus symbol override in `calculus_layout.py`

Expand `_calculus_symbol_override()`:

```python
OVERRIDE_RULES = [
    # (condition_fn, override_label, min_score)
    (lambda f: f["aspect_ratio"] < 0.65 and f["vertical_score"] > 0.15, "1", 0.82),
    (lambda f: f["num_components"] <= 1 and f["aspect_ratio"] > 2.0, "-", 0.84),
    # new:
    (lambda f: f["aspect_ratio"] < 0.8 and f["horizontal_score"] < 0.1, "d", 0.78),
    (lambda f: f["num_holes"] == 1 and 0.8 < f["aspect_ratio"] < 1.2, "0", 0.76),
]
```

### 4c. Calculus function prefix map in `calculus_rules.py`

Expand `_classify_function_prefix()` to cover more OCR error patterns:

```python
FUNCTION_PREFIX_MAP = {
    "sin": {"sin", "3/x", "5/x", "5(1", "51x", "5ix", "5/n", "s1n", "sln", "51n"},
    "cos": {"cos", "c05", "c0s", "co5", "c0S", "c05"},
    "tan": {"tan", "7an", "t4n", "t@n", "t4N"},
    "exp": {"exp", "6x2", "ex2", "64d", "e*p", "3xp", "e*p"},
    "log": {"ln", "1n", "1N", "lN", "lR"},
    "arcsin": {"arcsin", "arc5in", "arc51n"},
    "arccos": {"arccos", "arcco5", "arcc05"},
    "arctan": {"arctan", "arct4n", "arct@n"},
}
```

---

## Evaluation & Measurement

**File:** `tools/evaluate_samples.py` — add confusion matrix output.

### Changes
- After `_print_summary()`, output a confusion matrix: rows = true categories, columns = predicted categories for `auto` mode
- Add `--confusion-matrix` flag to print to stdout and save as CSV

### Success criteria
1. `--category auto` overall accuracy improves (baseline TBD from first run)
2. No single category accuracy drops by > 5% from baseline
3. Confusion matrix shows no systematic misrouting (e.g., all `printed_2d_layout` routed to `calculus`)
4. All 57 existing tests continue to pass

---

## Test Plan

### Updated tests
- `tests/test_auto_router.py`: Update assertions for new scoring-based routing (expected route hints stay the same; scoring values change)
- `tests/test_structure_2d.py`: No changes needed

### New tests
- `tests/test_structure_router_scoring.py`: Verify each scoring function produces expected relative ordering on known samples
- `tests/test_cross_validation.py`: Unit tests for `_cross_validate()` with mock candidates
- `tests/test_disambiguation_layers.py`: Verify Layer A/B/C rules don't conflict; test each confusion pair

---

## File Change Summary

| File | Change |
|---|---|
| `src/vision/structure_router.py` | Add 6 features fields + properties; replace `choose_route()` with scoring functions |
| `src/vision/auto_router.py` | Add `_cross_validate()`; modify `recognize_unknown()` flow for dual-path; calibrate `_CONFIDENCE_BASELINE` |
| `src/vision/recognizers/handwritten_rule_template.py` | Refactor `_disambiguate()` to 3-layer with `context` parameter; add confusion pair rules |
| `src/vision/calculus_rules.py` | Expand `_classify_function_prefix()` prefix map |
| `src/vision/calculus_layout.py` | Expand `_calculus_symbol_override()` rules |
| `tools/evaluate_samples.py` | Add confusion matrix output |
| `tests/test_auto_router.py` | Update for new scoring behavior |
| `tests/test_structure_router_scoring.py` | New — scoring function tests |
| `tests/test_cross_validation.py` | New — cross-validation unit tests |
| `tests/test_disambiguation_layers.py` | New — disambiguation layer tests |
