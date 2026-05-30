import unittest

from src.expression.serializer import normalize_expression_text_for_sympy, serialize_layout
from src.expression.types import LayoutNode


class SerializerTest(unittest.TestCase):
    def test_superscript_to_sympy(self):
        node = LayoutNode("superscript", children=[LayoutNode.symbol("x"), LayoutNode.symbol("2")])
        self.assertEqual(serialize_layout(node, target="sympy"), "x**2")

    def test_fraction_to_expression(self):
        node = LayoutNode("fraction", children=[LayoutNode.row([LayoutNode.symbol("1")]), LayoutNode.row([LayoutNode.symbol("2")])])
        self.assertEqual(serialize_layout(node), "1/2")

    def test_normalize_caret(self):
        self.assertEqual(normalize_expression_text_for_sympy("x^2"), "x**2")

    def test_derivative_text_to_sympy(self):
        self.assertEqual(normalize_expression_text_for_sympy("d/dx(x^2)"), "diff(x**2, x)")

    def test_indefinite_integral_text_to_sympy(self):
        self.assertEqual(normalize_expression_text_for_sympy("∫ x dx"), "integrate(x, x)")

    def test_definite_integral_text_to_sympy(self):
        self.assertEqual(normalize_expression_text_for_sympy("∫_0^1 x dx"), "integrate(x, (x, 0, 1))")

    def test_limit_text_to_sympy(self):
        self.assertEqual(
            normalize_expression_text_for_sympy("lim_{x->0} sin(x)/x"),
            "limit(sin(x)/x, x, 0)",
        )


if __name__ == "__main__":
    unittest.main()
