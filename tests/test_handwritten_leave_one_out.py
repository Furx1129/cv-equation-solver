import unittest
from pathlib import Path

from src.vision.normalization import normalize_formula_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image
from src.vision.recognizers.handwritten_rule_template import (
    HandwrittenRuleTemplateRecognizer,
    load_handwritten_templates,
    match_handwritten_symbol,
)
from src.vision.segmentation import segment_characters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDWRITTEN_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "handwritten_basic"
HANDWRITTEN_LABEL_DIR = PROJECT_ROOT / "data" / "labels" / "handwritten_basic"


class HandwrittenLeaveOneOutTest(unittest.TestCase):
    def test_excludes_current_sample_from_template_library(self):
        sample = "handwritten_0_001"
        library = load_handwritten_templates(
            PROJECT_ROOT / "data" / "samples" / "handwritten_basic",
            PROJECT_ROOT / "data" / "labels" / "handwritten_basic",
            exclude_stem=sample,
        )
        self.assertIn("0", library.images)
        self.assertEqual(len(library.images["0"]), 4)
        self.assertIn("0", library.features)
        self.assertEqual(len(library.features["0"]), 4)

    def test_historical_handwritten_confusions_are_fixed(self):
        cases = {
            "handwritten_1_002": "1",
            "handwritten_1_004": "1",
            "handwritten_2_001": "2",
            "handwritten_3_001": "3",
            "handwritten_3_002": "3",
            "handwritten_3_004": "3",
            "handwritten_3_005": "3",
            "handwritten_4_002": "4",
            "handwritten_5_002": "5",
            "handwritten_5_005": "5",
            "handwritten_6_002": "6",
            "handwritten_7_003": "7",
            "handwritten_8_002": "8",
            "handwritten_8_004": "8",
            "handwritten_9_004": "9",
            "handwritten_divide_001": "\u00f7",
            "handwritten_slash_004": "/",
            "handwritten_x_001": "x",
            "handwritten_x_003": "x",
            "handwritten_x_004": "x",
            "handwritten_y_001": "y",
            "handwritten_y_003": "y",
            "handwritten_y_005": "y",
        }
        for stem, expected in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(_recognize_single_leave_one_out(stem), expected)

    def test_open_three_in_test_sample_is_not_forced_to_eight(self):
        sample = PROJECT_ROOT / "data" / "samples" / "test" / "1.jpg"
        if not sample.exists():
            self.skipTest(f"sample not found: {sample}")
        result = HandwrittenRuleTemplateRecognizer().recognize(sample)
        self.assertEqual(result.expression_text, "3+5")


def _recognize_single_leave_one_out(stem: str) -> str:
    templates = load_handwritten_templates(
        HANDWRITTEN_SAMPLE_DIR,
        HANDWRITTEN_LABEL_DIR,
        exclude_stem=stem,
    )
    image = read_image(HANDWRITTEN_SAMPLE_DIR / f"{stem}.png")
    normalized = normalize_formula_image(image)
    preprocessed = preprocess_image(
        normalized.image,
        options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
    )
    segments = segment_characters(preprocessed.binary)
    label, _, _, _ = match_handwritten_symbol(segments[0].image, templates)
    return label


if __name__ == "__main__":
    unittest.main()
