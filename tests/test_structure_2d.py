import importlib.util
import unittest
from pathlib import Path

from src.vision.structure_2d import recognize_2d_layout


HAS_CV2 = importlib.util.find_spec("cv2") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class Structure2DTest(unittest.TestCase):
    def assert_recognizes(self, stem: str, expected: str):
        image = PROJECT_ROOT / "data" / "samples" / "printed_2d_layout" / f"{stem}.png"
        result = recognize_2d_layout(image)
        self.assertEqual(result.expression_text, expected)

    def test_simple_fraction(self):
        self.assert_recognizes("printed_2d_001", "1/2")

    def test_complex_fraction(self):
        self.assert_recognizes("printed_2d_004", "(x+1)/2")

    def test_superscript(self):
        self.assert_recognizes("printed_2d_005", "x^2")

    def test_subscript_row(self):
        self.assert_recognizes("printed_2d_008", "x_1+x_2")

    def test_subscript_superscript_combo(self):
        self.assert_recognizes("printed_2d_009", "a_1^2")

    def test_sqrt(self):
        self.assert_recognizes("printed_2d_010", "sqrt(9)")

    def test_sqrt_with_superscript(self):
        self.assert_recognizes("printed_2d_017", "sqrt(x^2+1)")


if __name__ == "__main__":
    unittest.main()
