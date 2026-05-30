import unittest

from src.expression.types import ExpressionResult
from src.solver.arithmetic import ArithmeticSolver, tokenize_arithmetic


class ArithmeticSolverTest(unittest.TestCase):
    def test_tokenize_multidigit_expression(self):
        self.assertEqual(tokenize_arithmetic("1*90-5"), ["1", "*", "90", "-", "5"])

    def test_solve_precedence(self):
        result = ArithmeticSolver().solve(ExpressionResult(text="1*90-5"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, 85)

    def test_solve_parentheses(self):
        result = ArithmeticSolver().solve(ExpressionResult(text="2*(3+4)"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, 14)

    def test_solve_display_multiply_and_divide_symbols(self):
        result = ArithmeticSolver().solve(ExpressionResult(text="2x(3+4)÷2"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, 7)

    def test_reject_unsupported_character(self):
        result = ArithmeticSolver().solve(ExpressionResult(text="2^3"))
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
