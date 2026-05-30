import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class LineSegmentationTest(unittest.TestCase):
    def test_segment_two_lines(self):
        import cv2
        import numpy as np

        from src.vision.line_segmentation import segment_lines

        image = np.full((180, 300), 255, dtype=np.uint8)
        cv2.putText(image, "1+2", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
        cv2.putText(image, "3+4", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
        lines = segment_lines(image, min_height=10, min_gap=10)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
