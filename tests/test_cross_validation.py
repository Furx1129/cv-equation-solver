from __future__ import annotations

import unittest

from src.expression.types import LayoutNode, RecognitionResult, SymbolToken
from src.vision.auto_router import RecognitionCandidate, _cross_validate


class CrossValidationTests(unittest.TestCase):
    def _make_result(self, texts: list[str], layout_type: str = "row", solver_answer: str = "") -> RecognitionResult:
        tokens = [SymbolToken(text=t, kind="symbol", bbox=(0, 0, 10, 10), confidence=0.9, source="test") for t in texts]
        layout = LayoutNode(node_type=layout_type, children=[], bbox=(0, 0, 0, 0))
        return RecognitionResult(tokens=tokens, expression_text="".join(texts), layout=layout)

    def _make_candidate(self, category: str, result: RecognitionResult, score: float = 0.65, solver_answer: str = "") -> RecognitionCandidate:
        return RecognitionCandidate(
            category=category, result=result, score=score, role="primary",
            solver_answer=solver_answer,
        )

    def test_identical_results_get_positive_score(self):
        r1 = self._make_result(["1", "+", "2"])
        r2 = self._make_result(["1", "+", "2"])
        c1 = self._make_candidate("printed_basic", r1, solver_answer="3")
        c2 = self._make_candidate("printed_2d_layout", r2, solver_answer="3")
        agree = _cross_validate(c1, c2)
        self.assertGreater(agree, 0.20)

    def test_contradictory_solver_results_get_negative(self):
        r1 = self._make_result(["1", "+", "2"])
        r2 = self._make_result(["1", "+", "2"])
        c1 = self._make_candidate("printed_basic", r1, solver_answer="3")
        c2 = self._make_candidate("printed_2d_layout", r2, solver_answer="4")
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
