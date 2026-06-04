from __future__ import annotations

import unittest

import numpy as np

from src.vision.recognizers.handwritten_rule_template import (
    TARGET_STROKE_PIXELS,
    align_segment_heights,
    refine_segment_binary,
    unify_stroke_fill_ratio,
)
from src.vision.template_matcher import _trim_foreground


class HandwrittenSegmentPrepTest(unittest.TestCase):
    def test_refine_segment_binary_produces_dark_on_white(self) -> None:
        canvas = np.full((40, 40), 255, dtype=np.uint8)
        canvas[10:30, 12:28] = 0
        binary = refine_segment_binary(canvas)
        self.assertEqual(binary.ndim, 2)
        self.assertGreater(int((binary < 128).sum()), 50)

    def test_unify_stroke_fill_ratio_enlarges_thin_strokes(self) -> None:
        binary = np.full((100, 100), 255, dtype=np.uint8)
        binary[45:55, 45:55] = 0
        before = int((_trim_foreground(binary) < 128).sum())
        unified = unify_stroke_fill_ratio(binary)
        after = int((_trim_foreground(unified) < 128).sum())
        self.assertGreater(after, before)

    def test_align_segment_heights_uses_median(self) -> None:
        small = np.full((30, 20), 255, dtype=np.uint8)
        small[8:22, 5:15] = 0
        large = np.full((60, 20), 255, dtype=np.uint8)
        large[5:55, 5:15] = 0
        aligned = align_segment_heights([small, large])
        heights = [np.where(image < 128)[0].size and (image < 128).any(axis=1).sum() for image in aligned]
        trimmed_heights = []
        for image in aligned:
            rows = np.where((image < 128).any(axis=1))[0]
            trimmed_heights.append(int(rows[-1] - rows[0] + 1) if rows.size else 0)
        self.assertEqual(trimmed_heights[0], trimmed_heights[1])


if __name__ == "__main__":
    unittest.main()
