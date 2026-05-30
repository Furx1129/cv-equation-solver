from __future__ import annotations

from dataclasses import dataclass

from src.expression.serializer import expression_to_sympy_text
from src.expression.types import ExpressionResult, SolveResult
from src.solver.base import Solver


@dataclass(frozen=True)
class SymbolicSolver(Solver):
    backend_name: str = "sympy"

    def solve(self, expression: ExpressionResult) -> SolveResult:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import convert_xor, standard_transformations, parse_expr
        except ImportError:
            return SolveResult(backend=self.backend_name, error="sympy is not installed")

        text = expression_to_sympy_text(expression)
        try:
            namespace = {
                "x": sp.Symbol("x"),
                "y": sp.Symbol("y"),
                "oo": sp.oo,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "log": sp.log,
                "exp": sp.exp,
                "sqrt": sp.sqrt,
                "integrate": sp.integrate,
                "diff": sp.diff,
                "limit": sp.limit,
                "Rational": sp.Rational,
                "pi": sp.pi,
                "E": sp.E,
                "e": sp.E,
            }
            parsed = parse_expr(
                text,
                local_dict=namespace,
                transformations=standard_transformations + (convert_xor,),
                evaluate=True,
            )
            simplified = sp.simplify(parsed)
            return SolveResult(answer=str(simplified), steps=[f"{text} => {simplified}"], backend=self.backend_name)
        except Exception as exc:
            return SolveResult(backend=self.backend_name, error=str(exc))
