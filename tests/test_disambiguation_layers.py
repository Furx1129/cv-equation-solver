from __future__ import annotations

import unittest

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

    def test_balanced_diagonal_forces_x(self):
        candidates = {"x": 0.52, "×": 0.54, "y": 0.48}
        f = self._features(
            diag_pos_score=0.35, diag_neg_score=0.35, horizontal_score=0.05,
            aspect_ratio=0.95, centroid_x=0.50,
        )
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "x")

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

    def test_confusion_pair_zero_vs_oh(self):
        candidates = {"0": 0.45, "o": 0.48, "O": 0.44}
        f = self._features(aspect_ratio=0.90, num_holes=1)
        adjusted, reason = _disambiguate(f, candidates)
        best = max(adjusted, key=adjusted.get)
        self.assertEqual(best, "o")


if __name__ == "__main__":
    unittest.main()
