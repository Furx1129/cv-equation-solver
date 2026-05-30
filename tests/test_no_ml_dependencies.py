import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("tensorflow", "torch", "keras", "sklearn", "easyocr", "pytesseract", "paddleocr", "onnx")


class NoMLDependenciesTest(unittest.TestCase):
    def test_runtime_code_does_not_import_ml_or_ocr_packages(self):
        files = list((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "run.py", PROJECT_ROOT / "requirements.txt"]
        for path in files:
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            for name in FORBIDDEN:
                self.assertNotIn(name, lowered, f"{name} found in {path}")

    def test_runtime_code_does_not_use_eval(self):
        files = list((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "run.py"]
        for path in files:
            self.assertNotIn("eval(", path.read_text(encoding="utf-8"), str(path))


if __name__ == "__main__":
    unittest.main()
