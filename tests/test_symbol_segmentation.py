import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
class SymbolSegmentationTest(unittest.TestCase):
    def test_segment_symbols_bbox_in_bounds(self):
        import cv2
        import numpy as np

        from src.vision.symbol_segmentation import segment_symbols

        image = np.full((100, 260), 255, dtype=np.uint8)
        cv2.putText(image, "1+2", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 0, 4)
        regions = segment_symbols(image, min_area=5)
        self.assertGreaterEqual(len(regions), 3)
        for region in regions:
            x, y, w, h = region.bbox
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, image.shape[1])
            self.assertLessEqual(y + h, image.shape[0])


if __name__ == "__main__":
    unittest.main()
