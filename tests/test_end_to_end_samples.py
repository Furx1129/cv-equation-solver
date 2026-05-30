import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EndToEndSampleDataTest(unittest.TestCase):
    def test_all_samples_have_matching_labels(self):
        sample_root = PROJECT_ROOT / "data" / "samples"
        label_root = PROJECT_ROOT / "data" / "labels"
        unlabeled_diagnostic_dirs = {"task1", "test"}
        for sample_dir in sample_root.iterdir():
            if not sample_dir.is_dir() or sample_dir.name in unlabeled_diagnostic_dirs:
                continue
            label_dir = label_root / sample_dir.name
            self.assertTrue(label_dir.exists(), sample_dir.name)
            for image in sample_dir.iterdir():
                if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                self.assertTrue((label_dir / f"{image.stem}.txt").exists(), image.name)


if __name__ == "__main__":
    unittest.main()
