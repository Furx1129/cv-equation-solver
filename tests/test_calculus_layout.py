import unittest
from pathlib import Path

from src.vision.calculus_layout import recognize_calculus_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALCULUS_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "calculus"


class CalculusLayoutTest(unittest.TestCase):
    def test_calculus_samples_recognize_to_sympy_text(self):
        cases = {
            "calculus_001": ("integrate(x, x)", "integral"),
            "calculus_002": ("integrate(x**2, x)", "integral"),
            "calculus_003": ("integrate(sin(x), x)", "integral"),
            "calculus_004": ("diff(x**2, x)", "derivative"),
            "calculus_005": ("diff(sin(x), x)", "derivative"),
            "calculus_006": ("diff(exp(x), x)", "derivative"),
            "calculus_007": ("limit(sin(x)/x, x, 0)", "limit"),
            "calculus_008": ("limit((x**2-1)/(x-1), x, 1)", "limit"),
            "calculus_009": ("limit(1/x, x, oo)", "limit"),
            "calculus_010": ("integrate(1/x, x)", "integral"),
        }
        for stem, (expected, layout_type) in cases.items():
            with self.subTest(stem=stem):
                result = recognize_calculus_layout(CALCULUS_SAMPLE_DIR / f"{stem}.png")
                self.assertEqual(result.sympy_text, expected)
                self.assertIsNotNone(result.layout)
                self.assertEqual(result.layout.node_type, layout_type)


if __name__ == "__main__":
    unittest.main()
