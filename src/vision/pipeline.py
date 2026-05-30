from pathlib import Path

from src.expression.types import RecognitionResult
from src.vision.auto_router import recognize_unknown
from src.vision.calculus_layout import recognize_calculus_layout
from src.vision.recognizers.handwritten_rule_template import HandwrittenRuleTemplateRecognizer
from src.vision.recognizers.printed_template import DEFAULT_TEMPLATE_DIR, PrintedTemplateRecognizer


def recognize_image(
    image_path: str | Path,
    debug_dir: str | Path | None = None,
    backend: str = "printed",
) -> RecognitionResult:
    if backend == "printed":
        return PrintedTemplateRecognizer().recognize(image_path=image_path, debug_dir=debug_dir)
    if backend == "handwritten":
        return HandwrittenRuleTemplateRecognizer().recognize(image_path=image_path, debug_dir=debug_dir)
    if backend == "calculus":
        return recognize_calculus_layout(image_path=image_path, debug_dir=debug_dir)
    if backend == "auto":
        decision = recognize_unknown(image_path=image_path, debug_dir=debug_dir)
        return decision.selected.result or RecognitionResult(tokens=[], expression_text="")
    raise ValueError(f"unknown recognizer backend: {backend}")


def _recognition_penalty(result: RecognitionResult) -> tuple[int, int, float]:
    unknown_count = sum(1 for token in result.tokens if token.text == "UNKNOWN")
    low_confidence = sum(1 for token in result.tokens if token.confidence < 0.55)
    average_confidence = sum(token.confidence for token in result.tokens) / max(1, len(result.tokens))
    return (unknown_count, low_confidence, -average_confidence)
