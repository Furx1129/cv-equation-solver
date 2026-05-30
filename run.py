from __future__ import annotations

import argparse
from pathlib import Path

from src.expression.types import ExpressionResult
from src.expression.normalizer import normalize_tokens
from src.solver.arithmetic import ArithmeticSolver
from src.solver.symbolic import SymbolicSolver
from src.vision.pipeline import recognize_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize a printed equation image and solve it.")
    parser.add_argument("image", help="Path to the equation image.")
    parser.add_argument("--backend", choices=["auto", "arithmetic", "symbolic"], default="auto")
    parser.add_argument("--recognizer", choices=["printed", "handwritten", "calculus", "auto"], default="printed")
    parser.add_argument("--debug-dir", help="Optional directory for binary image, segments, and token table.")
    args = parser.parse_args()

    recognition = recognize_image(Path(args.image), debug_dir=args.debug_dir, backend=args.recognizer)
    expression = normalize_tokens(recognition.tokens)
    backend = args.backend
    if backend == "auto":
        backend = "symbolic" if recognition.sympy_text or _needs_symbolic(expression.text) else "arithmetic"
    if backend == "symbolic" and recognition.sympy_text:
        expression = ExpressionResult(text=recognition.sympy_text, tokens=recognition.tokens, layout=recognition.layout)
    solver = SymbolicSolver() if backend == "symbolic" else ArithmeticSolver()
    solve_result = solver.solve(expression)

    print(f"Expression: {recognition.expression_text}")
    if solve_result.error:
        print(f"Error: {solve_result.error}")
    else:
        print(f"Answer: {solve_result.answer}")

    print("Tokens:")
    for index, token in enumerate(recognition.tokens):
        print(
            f"  {index:02d}: {token.text} "
            f"score={token.confidence:.3f} bbox={token.bbox} source={token.source}"
        )

    if recognition.warnings:
        print("Warnings:")
        for warning in recognition.warnings:
            print(f"  - {warning}")

    if recognition.debug_artifacts:
        print("Debug artifacts:")
        for name, path in recognition.debug_artifacts.items():
            print(f"  {name}: {path}")

    return 1 if solve_result.error else 0


def _needs_symbolic(expression: str) -> bool:
    if set(expression) <= set("0123456789.+-*/x×÷脳梅()= "):
        return False
    symbolic_markers = set("y^")
    return any(char in symbolic_markers for char in expression) or any(
        name in expression for name in ("sqrt", "integrate", "diff", "limit")
    )


if __name__ == "__main__":
    raise SystemExit(main())
