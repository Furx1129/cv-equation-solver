import importlib.util
import unittest

from src.expression.types import ExpressionResult
from src.solver.symbolic import SymbolicSolver

HAS_SYMPY = importlib.util.find_spec("sympy") is not None


@unittest.skipUnless(HAS_SYMPY, "sympy is not installed")
class SymbolicSolverTest(unittest.TestCase):
    def test_integrate(self):
        result = SymbolicSolver().solve(ExpressionResult(text="integrate(x, x)"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "x**2/2")

    def test_diff(self):
        result = SymbolicSolver().solve(ExpressionResult(text="diff(x^2, x)"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "2*x")

    def test_definite_integral(self):
        result = SymbolicSolver().solve(ExpressionResult(text="∫_0^1 x dx"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "1/2")

    def test_limit(self):
        result = SymbolicSolver().solve(ExpressionResult(text="lim_{x->0} sin(x)/x"))
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "1")


if __name__ == "__main__":
    unittest.main()
