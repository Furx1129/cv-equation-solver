from __future__ import annotations

import unittest
from pathlib import Path

from src.vision.auto_router import extract_image_structure_features, recognize_unknown
from src.vision.structure_router import analyze_formula_structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AutoRouterTests(unittest.TestCase):
    def test_flat_printed_sample_skips_complex_branches(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "printed_basic" / "printed_basic_007.png"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        rejected = {candidate.category: candidate.reject_stage for candidate in decision.candidates}
        self.assertIn(decision.structure_analysis.route_hint, {"printed_basic", "printed_decimal_negative"})
        self.assertEqual(decision.selected.category, "printed_basic")
        self.assertIn(rejected.get("calculus"), {"fallback_not_needed", "route_prior"})

    def test_structural_sample_routes_to_2d_before_recognition(self) -> None:
        image = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_001.png"
        decision = recognize_unknown(image, solver_timeout_ms=1000)
        self.assertEqual(decision.structure_analysis.route_hint, "printed_2d_layout")
        self.assertEqual(decision.selected.category, "printed_2d_layout")
        calculus = next(candidate for candidate in decision.candidates if candidate.category == "calculus")
        self.assertIn(calculus.reject_stage, {"fallback_not_needed", "route_prior"})

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


if __name__ == "__main__":
    unittest.main()
