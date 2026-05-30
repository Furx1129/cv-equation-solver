import unittest

from src.expression.display_normalizer import (
    normalize_arithmetic_for_solver,
    normalize_for_comparison,
    normalize_for_display,
)


class DisplayNormalizerTest(unittest.TestCase):
    def test_arithmetic_multiplication_variants_display_as_x(self):
        expected = "2x(3+4)=14"
        self.assertEqual(normalize_for_display("2×(3+4)=14", "printed_basic"), expected)
        self.assertEqual(normalize_for_display("2*(3+4)=14", "printed_basic"), expected)
        self.assertEqual(normalize_for_display("2脳(3+4)=14", "printed_basic"), expected)

    def test_arithmetic_division_variants_display_as_divide_symbol(self):
        expected = "8÷2=4"
        self.assertEqual(normalize_for_display("8/2=4", "printed_decimal_negative"), expected)
        self.assertEqual(normalize_for_display("8÷2=4", "printed_decimal_negative"), expected)
        self.assertEqual(normalize_for_display("8梅2=4", "printed_decimal_negative"), expected)

    def test_non_arithmetic_preserves_variable_x(self):
        self.assertEqual(normalize_for_display("x^2", "calculus"), "x^2")
        self.assertEqual(normalize_for_comparison("x^2", "printed_2d_layout"), "x^2")

    def test_arithmetic_solver_uses_internal_operators(self):
        self.assertEqual(normalize_arithmetic_for_solver("2x(3+4)÷2"), "2*(3+4)/2")


if __name__ == "__main__":
    unittest.main()
