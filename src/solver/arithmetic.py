from __future__ import annotations

from dataclasses import dataclass

from src.expression.display_normalizer import normalize_arithmetic_for_solver
from src.expression.types import ExpressionResult, SolveResult
from src.solver.base import Solver


Token = str


class ArithmeticError(ValueError):
    pass


@dataclass
class _Parser:
    tokens: list[Token]
    position: int = 0
    steps: list[str] | None = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []

    def parse(self) -> float:
        if not self.tokens:
            raise ArithmeticError("empty expression")
        value = self._expression()
        if self._peek() is not None:
            raise ArithmeticError(f"unexpected token: {self._peek()}")
        return value

    def _expression(self) -> float:
        value = self._term()
        while self._peek() in {"+", "-"}:
            op = self._consume()
            right = self._term()
            left = value
            value = left + right if op == "+" else left - right
            self.steps.append(f"{_format_number(left)} {op} {_format_number(right)} = {_format_number(value)}")
        return value

    def _term(self) -> float:
        value = self._factor()
        while self._peek() in {"*", "/"}:
            op = self._consume()
            right = self._factor()
            left = value
            if op == "/" and right == 0:
                raise ArithmeticError("division by zero")
            value = left * right if op == "*" else left / right
            self.steps.append(f"{_format_number(left)} {op} {_format_number(right)} = {_format_number(value)}")
        return value

    def _factor(self) -> float:
        token = self._peek()
        if token is None:
            raise ArithmeticError("unexpected end of expression")
        if token == "(":
            self._consume()
            value = self._expression()
            if self._peek() != ")":
                raise ArithmeticError("missing closing parenthesis")
            self._consume()
            return value
        if token == "-":
            self._consume()
            return -self._factor()
        if _is_number(token):
            self._consume()
            return float(token)
        raise ArithmeticError(f"unexpected token: {token}")

    def _peek(self) -> Token | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def _consume(self) -> Token:
        token = self.tokens[self.position]
        self.position += 1
        return token


class ArithmeticSolver(Solver):
    backend_name = "local_arithmetic"

    def solve(self, expression: ExpressionResult) -> SolveResult:
        try:
            text = normalize_arithmetic_for_solver(expression.text)
            if "=" in text:
                left_text, right_text = _split_equality(text)
                left = _Parser(tokens=tokenize_arithmetic(left_text)).parse()
                right = _Parser(tokens=tokenize_arithmetic(right_text)).parse()
                answer = left == right
                steps = [f"{left_text} = {_format_number(left)}", f"{right_text} = {_format_number(right)}"]
                return SolveResult(answer=answer, steps=steps, backend=self.backend_name, error=None)
            tokens = tokenize_arithmetic(text)
            parser = _Parser(tokens=tokens)
            answer = parser.parse()
            return SolveResult(
                answer=_normalize_answer(answer),
                steps=parser.steps or [],
                backend=self.backend_name,
                error=None,
            )
        except ArithmeticError as exc:
            return SolveResult(answer=None, steps=[], backend=self.backend_name, error=str(exc))


def tokenize_arithmetic(expression: str) -> list[Token]:
    expression = normalize_arithmetic_for_solver(expression)
    allowed = set("0123456789.+-*/() ")
    invalid = sorted(set(expression) - allowed)
    if invalid:
        raise ArithmeticError(f"unsupported characters: {''.join(invalid)}")

    tokens: list[Token] = []
    number = []
    for char in expression:
        if char.isspace():
            continue
        if char.isdigit() or char == ".":
            number.append(char)
            continue
        if number:
            tokens.append(_validate_number("".join(number)))
            number.clear()
        tokens.append(char)
    if number:
        tokens.append(_validate_number("".join(number)))
    return tokens


def _validate_number(text: str) -> str:
    if text.count(".") > 1 or text == ".":
        raise ArithmeticError(f"invalid number: {text}")
    return text


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_answer(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return value


def _format_number(value: float) -> str:
    normalized = _normalize_answer(value)
    return str(normalized)


def _split_equality(expression: str) -> tuple[str, str]:
    parts = expression.split("=")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ArithmeticError("only one complete equality is supported")
    return parts[0], parts[1]
