import importlib.util
import unittest
from pathlib import Path


HAS_CV2 = importlib.util.find_spec("cv2") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates" / "printed_basic"
EXPECTED_TEMPLATE_FILES = {
    "tem_0.jpg",
    "tem_1.jpg",
    "tem_2.jpg",
    "tem_3.jpg",
    "tem_4.jpg",
    "tem_5.jpg",
    "tem_6.jpg",
    "tem_7.jpg",
    "tem_8.jpg",
    "tem_9.jpg",
    "tem_plus.jpg",
    "tem_minus.jpg",
    "tem_times.jpg",
    "tem_divide.jpg",
    "tem_left.jpg",
    "tem_right.jpg",
    "tem_0_gen.png",
    "tem_1_gen.png",
    "tem_2_gen.png",
    "tem_3_gen.png",
    "tem_4_gen.png",
    "tem_5_gen.png",
    "tem_6_gen.png",
    "tem_7_gen.png",
    "tem_8_gen.png",
    "tem_9_gen.png",
    "tem_plus_gen.png",
    "tem_minus_gen.png",
    "tem_left_gen.png",
    "tem_right_gen.png",
    "tem_equal.png",
    "tem_dot.png",
    "tem_mul_x.png",
    "tem_divide_symbol.png",
    "tem_var_x.png",
    "tem_var_y.png",
    "tem_caret.png",
    "tem_sqrt.png",
    "tem_integral.png",
    "tem_d.png",
    "tem_partial.png",
    "tem_arrow.png",
    "tem_infinity.png",
    "tem_l.png",
    "tem_i.png",
    "tem_m.png",
    "tem_s.png",
    "tem_n.png",
    "tem_c.png",
    "tem_o.png",
    "tem_t.png",
    "tem_a.png",
    "tem_e.png",
    "tem_p.png",
    "tem_g.png",
    "tem_r.png",
    "tem_q.png",
    "tem_lim.png",
    "tem_dx.png",
    "tem_d_over_dx.png",
}


class PipelineResourceTest(unittest.TestCase):
    def test_template_assets_are_absorbed(self):
        for filename in EXPECTED_TEMPLATE_FILES:
            self.assertTrue((DEFAULT_TEMPLATE_DIR / filename).exists(), filename)

    @unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
    def test_template_loader_reads_all_templates(self):
        from src.vision.template_matcher import TEMPLATE_FILES
        from src.vision.template_matcher import load_templates

        templates = load_templates(DEFAULT_TEMPLATE_DIR)
        self.assertEqual(set(templates), set(TEMPLATE_FILES))
        self.assertIn(".", templates)
        self.assertIn("=", templates)
        self.assertIn("x", templates)
        self.assertIn("÷", templates)

    @unittest.skipUnless(HAS_CV2, "opencv-python is not installed")
    def test_eq2_pipeline_stays_independent_from_task1(self):
        from src.expression.normalizer import normalize_tokens
        from src.solver.arithmetic import ArithmeticSolver
        from src.vision.pipeline import DEFAULT_TEMPLATE_DIR as PIPELINE_TEMPLATE_DIR
        from src.vision.pipeline import recognize_image

        image_path = PROJECT_ROOT / "data" / "samples" / "task1" / "eq2.jpg"
        if not image_path.exists():
            self.skipTest("legacy absorbed task1 sample is not present")
        recognition = recognize_image(image_path)
        self.assertNotIn("task1\\template", str(PIPELINE_TEMPLATE_DIR))
        self.assertIn(recognition.expression_text, {"1*90-5", "1x90-5"})
        solved = ArithmeticSolver().solve(normalize_tokens(recognition.tokens))
        self.assertEqual(solved.answer, 85)


if __name__ == "__main__":
    unittest.main()
