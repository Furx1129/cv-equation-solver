import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class PreprocessTest(unittest.TestCase):
    def test_otsu_and_adaptive_produce_binary_images(self):
        import cv2
        import numpy as np

        from src.vision.preprocess import PreprocessOptions, preprocess_image

        image = np.full((80, 220), 255, dtype=np.uint8)
        cv2.putText(image, "7+8", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        for method in ("otsu", "adaptive"):
            result = preprocess_image(image, options=PreprocessOptions(threshold_method=method, median_kernel=0))
            values = set(np.unique(result.binary).tolist())
            self.assertTrue(values.issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
