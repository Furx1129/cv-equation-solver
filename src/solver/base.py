from __future__ import annotations

from abc import ABC, abstractmethod

from src.expression.types import ExpressionResult, SolveResult


class Solver(ABC):
    backend_name: str

    @abstractmethod
    def solve(self, expression: ExpressionResult) -> SolveResult:
        raise NotImplementedError
