import importlib.util
import unittest
from pathlib import Path

from src.expression.serializer import serialize_layout
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image
from src.vision.recognizers.handwritten_rule_template import HandwrittenRuleTemplateRecognizer
from src.vision.structure_2d import (
    _crop_binary,
    looks_like_2d_structure,
    looks_like_handwritten_2d_structure,
    recognize_2d_layout,
)


HAS_CV2 = importlib.util.find_spec("cv2") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class Handwritten2DStructureTest(unittest.TestCase):
    def test_looks_like_2d_on_printed_fraction_sample(self):
        image = read_image(PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_001.png")
        preprocessed = preprocess_image(
            image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
        )
        self.assertTrue(looks_like_2d_structure(_crop_binary(preprocessed.binary)))

    def test_looks_like_2d_on_printed_superscript_sample(self):
        image = read_image(PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_005.png")
        preprocessed = preprocess_image(
            image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
        )
        self.assertTrue(looks_like_2d_structure(_crop_binary(preprocessed.binary)))

    def test_flat_handwritten_basic_not_flagged_as_2d(self):
        image = read_image(PROJECT_ROOT / "data" / "samples" / "handwritten_basic" / "handwritten_0_001.png")
        preprocessed = preprocess_image(
            image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
        )
        self.assertFalse(looks_like_2d_structure(_crop_binary(preprocessed.binary)))

    def test_handwritten_gate_only_fraction_not_superscript(self):
        image = read_image(PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_005.png")
        preprocessed = preprocess_image(
            image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
        )
        cropped = _crop_binary(preprocessed.binary)
        self.assertTrue(looks_like_2d_structure(cropped))
        self.assertFalse(looks_like_handwritten_2d_structure(cropped))

    def test_handwritten_superscript_stays_on_flat_pipeline(self):
        image_path = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_005.png"
        result = HandwrittenRuleTemplateRecognizer().recognize(image_path)
        sources = {token.source for token in result.tokens}
        self.assertNotIn("handwritten_2d_structure", sources)
        self.assertTrue(all(source == "handwritten_rule_template" for source in sources))

    def test_handwritten_recognizer_uses_2d_parser_on_fraction_layout(self):
        image_path = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_001.png"
        result = HandwrittenRuleTemplateRecognizer().recognize(image_path)
        sources = {token.source for token in result.tokens}
        self.assertIn("handwritten_2d_structure", sources)
        self.assertEqual(result.layout.node_type if result.layout else None, "fraction")
        self.assertTrue(any("used handwritten 2d structure parser" in warning for warning in result.warnings))

    def test_printed_2d_layout_still_works(self):
        image_path = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / "printed_2d_005.png"
        result = recognize_2d_layout(image_path)
        self.assertEqual(result.expression_text, "x^2")

    def test_flat_handwritten_expression_uses_layout_for_superscript(self):
        from src.expression.types import SymbolToken

        tokens = [
            SymbolToken("2", "digit", (10, 40, 30, 50), 0.9, "handwritten_rule_template"),
            SymbolToken("2", "digit", (45, 5, 18, 20), 0.9, "handwritten_rule_template"),
        ]
        from src.vision.layout_analysis import analyze_layout

        layout = analyze_layout(tokens)
        text = serialize_layout(layout, target="plain")
        self.assertEqual(text, "2^2")


if __name__ == "__main__":
    unittest.main()
