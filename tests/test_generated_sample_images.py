import unittest
from pathlib import Path

from tools.generate_sample_images import SAMPLES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GeneratedSampleImagesTest(unittest.TestCase):
    def test_generated_sample_specs_have_images_and_labels(self):
        sample_root = PROJECT_ROOT / "data" / "samples"
        label_root = PROJECT_ROOT / "data" / "labels"
        for spec in SAMPLES:
            with self.subTest(stem=spec.stem):
                image_path = sample_root / spec.category / f"{spec.stem}.png"
                label_path = label_root / spec.category / f"{spec.stem}.txt"
                self.assertTrue(image_path.exists(), image_path)
                self.assertTrue(label_path.exists(), label_path)
                self.assertEqual(label_path.read_text(encoding="utf-8"), spec.label)


if __name__ == "__main__":
    unittest.main()
