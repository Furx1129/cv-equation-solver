import unittest

from src.expression.types import SymbolToken
from src.vision.layout_analysis import analyze_layout


class LayoutAnalysisTest(unittest.TestCase):
    def test_detect_simple_fraction(self):
        tokens = [
            SymbolToken("1", "digit", (20, 0, 10, 20), 1.0, "test"),
            SymbolToken("-", "operator", (0, 30, 60, 3), 1.0, "test"),
            SymbolToken("2", "digit", (20, 45, 10, 20), 1.0, "test"),
        ]
        layout = analyze_layout(tokens)
        self.assertEqual(layout.node_type, "fraction")

    def test_detect_superscript(self):
        tokens = [
            SymbolToken("x", "symbol", (0, 20, 20, 30), 1.0, "test"),
            SymbolToken("2", "digit", (24, 0, 10, 12), 1.0, "test"),
        ]
        layout = analyze_layout(tokens)
        self.assertEqual(layout.children[0].node_type, "superscript")

    def test_detect_integral_layout_from_tokens(self):
        tokens = [
            SymbolToken("∫", "symbol", (0, 0, 10, 30), 1.0, "test"),
            SymbolToken("_", "operator", (12, 24, 4, 4), 1.0, "test"),
            SymbolToken("0", "digit", (16, 24, 6, 8), 1.0, "test"),
            SymbolToken("^", "operator", (12, 0, 4, 4), 1.0, "test"),
            SymbolToken("1", "digit", (16, 0, 6, 8), 1.0, "test"),
            SymbolToken("x", "symbol", (28, 12, 10, 12), 1.0, "test"),
            SymbolToken("d", "symbol", (42, 12, 10, 12), 1.0, "test"),
            SymbolToken("x", "symbol", (54, 12, 10, 12), 1.0, "test"),
        ]
        layout = analyze_layout(tokens)
        self.assertEqual(layout.node_type, "integral")
        self.assertEqual(layout.metadata["lower"], "0")
        self.assertEqual(layout.metadata["upper"], "1")

    def test_detect_limit_layout_from_tokens(self):
        tokens = [
            SymbolToken("l", "symbol", (0, 0, 5, 10), 1.0, "test"),
            SymbolToken("i", "symbol", (6, 0, 5, 10), 1.0, "test"),
            SymbolToken("m", "symbol", (12, 0, 10, 10), 1.0, "test"),
            SymbolToken("_", "operator", (24, 10, 4, 4), 1.0, "test"),
            SymbolToken("{", "paren", (30, 10, 4, 8), 1.0, "test"),
            SymbolToken("x", "symbol", (36, 10, 8, 8), 1.0, "test"),
            SymbolToken("->", "operator", (46, 10, 10, 8), 1.0, "test"),
            SymbolToken("0", "digit", (58, 10, 8, 8), 1.0, "test"),
            SymbolToken("}", "paren", (68, 10, 4, 8), 1.0, "test"),
            SymbolToken("x", "symbol", (78, 0, 8, 10), 1.0, "test"),
        ]
        layout = analyze_layout(tokens)
        self.assertEqual(layout.node_type, "limit")
        self.assertEqual(layout.metadata["target"], "0")

    def test_detect_sqrt_layout(self):
        tokens = [
            SymbolToken("√", "symbol", (0, 0, 15, 20), 1.0, "test"),
            SymbolToken("x", "symbol", (16, 5, 10, 10), 1.0, "test"),
        ]
        layout = analyze_layout(tokens)
        self.assertEqual(layout.node_type, "sqrt")


if __name__ == "__main__":
    unittest.main()
