import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class DeskewTest(unittest.TestCase):
    def test_deskew_returns_reasonable_angle_for_rotated_text(self):
        import cv2
        import numpy as np

        from src.vision.deskew import deskew_image, rotate_image

        image = np.full((120, 300), 255, dtype=np.uint8)
        cv2.putText(image, "123", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
        rotated = rotate_image(image, -8)
        _, binary = cv2.threshold(rotated, 128, 255, cv2.THRESH_BINARY)
        result = deskew_image(binary)
        self.assertLessEqual(abs(result.angle), 20)
        self.assertEqual(result.image.ndim, 2)


if __name__ == "__main__":
    unittest.main()
