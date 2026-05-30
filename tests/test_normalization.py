import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class NormalizationTest(unittest.TestCase):
    def test_crop_large_whitespace_and_preserve_height(self):
        import cv2
        import numpy as np

        from src.vision.normalization import normalize_formula_image

        image = np.full((300, 1000), 255, dtype=np.uint8)
        cv2.putText(image, "12+3", (380, 165), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5, cv2.LINE_AA)
        result = normalize_formula_image(image, target_height=128, margin=12)
        self.assertEqual(result.image.shape[0], 128)
        self.assertLess(result.output_size[0], image.shape[1])
        self.assertGreater(result.scale, 0)


if __name__ == "__main__":
    unittest.main()
