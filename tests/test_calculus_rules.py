import unittest

from src.vision.calculus_rules import apply_calculus_geometry_rules
from src.expression.types import RecognitionResult, SymbolToken


class CalculusRulesTest(unittest.TestCase):
    def test_rewrites_integral_dx_pattern(self):
        result = RecognitionResult(
            tokens=[
                SymbolToken("1", "digit", (0, 0, 30, 120), 0.4, "test"),
                SymbolToken("x", "symbol", (40, 50, 30, 30), 0.8, "test"),
                SymbolToken("y", "symbol", (80, 20, 30, 70), 0.4, "test"),
                SymbolToken("x", "symbol", (120, 50, 30, 30), 0.8, "test"),
            ],
            expression_text="1xyx",
        )
        fixed = apply_calculus_geometry_rules(result)
        self.assertEqual(fixed.expression_text, "integrate(x, x)")
        self.assertEqual(fixed.sympy_text, "integrate(x, x)")

    def test_detects_derivative_operator_region(self):
        result = RecognitionResult(
            tokens=[
                SymbolToken("6", "digit", (0, 0, 120, 160), 0.3, "test"),
                SymbolToken("x", "symbol", (140, 80, 40, 50), 0.8, "test"),
                SymbolToken("2", "digit", (185, 30, 20, 20), 0.8, "test"),
            ],
            expression_text="6x2",
        )
        fixed = apply_calculus_geometry_rules(result)
        self.assertEqual(fixed.sympy_text, "diff(x**2, x)")

    def test_recovers_common_handwritten_sin_prefix(self):
        result = RecognitionResult(
            tokens=[
                SymbolToken("1", "digit", (0, 0, 30, 120), 0.4, "test"),
                SymbolToken("3", "digit", (40, 50, 20, 30), 0.7, "test"),
                SymbolToken("/", "operator", (65, 30, 15, 50), 0.5, "test"),
                SymbolToken("x", "symbol", (85, 50, 20, 30), 0.5, "test"),
                SymbolToken("(", "paren", (110, 30, 15, 60), 0.8, "test"),
                SymbolToken("x", "symbol", (130, 50, 20, 30), 0.8, "test"),
                SymbolToken(")", "paren", (155, 30, 15, 60), 0.8, "test"),
                SymbolToken("y", "symbol", (190, 20, 30, 70), 0.4, "test"),
                SymbolToken("x", "symbol", (230, 50, 30, 30), 0.8, "test"),
            ],
            expression_text="13/x(x)yx",
        )
        fixed = apply_calculus_geometry_rules(result)
        self.assertEqual(fixed.sympy_text, "integrate(sin(x), x)")


if __name__ == "__main__":
    unittest.main()
