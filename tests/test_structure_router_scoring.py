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
            multi_scale_edge_ratio=1.1, top_alignment_score=0.5,
        )

    def _fraction_2d(self) -> ImageStructureFeatures:
        return ImageStructureFeatures(
            width=120, height=180, aspect_ratio=0.67,
            foreground_components=4, left_tall_components=0,
            long_horizontal_lines=1, fraction_like_lines=1,
            stacked_component_pairs=2, edge_roughness=0.10,
            foreground_fill_ratio=0.15,
            vertical_symmetry=0.72, component_height_variance=0.10,
            horizontal_run_count=3, left_right_density_ratio=1.0,
            multi_scale_edge_ratio=1.2, top_alignment_score=0.4,
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
            width=200, height=100, aspect_ratio=2.0,
            foreground_components=8, left_tall_components=1,
            long_horizontal_lines=0, fraction_like_lines=1,
            stacked_component_pairs=2, edge_roughness=0.28,
            foreground_fill_ratio=0.10,
            vertical_symmetry=0.45, component_height_variance=0.25,
            horizontal_run_count=3, left_right_density_ratio=1.8,
            multi_scale_edge_ratio=1.9, top_alignment_score=0.0,
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
