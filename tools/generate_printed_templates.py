from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates" / "printed_basic"
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\ARIALUNI.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\symbol.ttf"),
)

TEMPLATES = {
    "tem_0_gen.png": "0",
    "tem_1_gen.png": "1",
    "tem_2_gen.png": "2",
    "tem_3_gen.png": "3",
    "tem_4_gen.png": "4",
    "tem_5_gen.png": "5",
    "tem_6_gen.png": "6",
    "tem_7_gen.png": "7",
    "tem_8_gen.png": "8",
    "tem_9_gen.png": "9",
    "tem_plus_gen.png": "+",
    "tem_minus_gen.png": "-",
    "tem_left_gen.png": "(",
    "tem_right_gen.png": ")",
    "tem_equal.png": "=",
    "tem_dot.png": ".",
    "tem_mul_x.png": "x",
    "tem_divide_symbol.png": "÷",
    "tem_var_x.png": "x",
    "tem_var_y.png": "y",
    "tem_caret.png": "^",
    "tem_sqrt.png": "√",
    "tem_integral.png": "∫",
    "tem_d.png": "d",
    "tem_partial.png": "∂",
    "tem_arrow.png": "→",
    "tem_infinity.png": "∞",
    "tem_l.png": "l",
    "tem_i.png": "i",
    "tem_m.png": "m",
    "tem_s.png": "s",
    "tem_n.png": "n",
    "tem_c.png": "c",
    "tem_o.png": "o",
    "tem_t.png": "t",
    "tem_a.png": "a",
    "tem_e.png": "e",
    "tem_p.png": "p",
    "tem_g.png": "g",
    "tem_r.png": "r",
    "tem_q.png": "q",
    "tem_lim.png": "lim",
    "tem_dx.png": "dx",
    "tem_d_over_dx.png": "d/dx",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixed printed symbol templates.")
    parser.add_argument("--output-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--force", action="store_true", help="Overwrite templates that already exist.")
    parser.add_argument("--only", help="Generate only one template filename, for example tem_d_over_dx.png.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_font()

    written = []
    skipped = []
    for filename, text in TEMPLATES.items():
        if args.only is not None and filename != args.only:
            continue
        path = output_dir / filename
        if path.exists() and not args.force:
            skipped.append(filename)
            continue
        if filename == "tem_d_over_dx.png":
            _render_d_over_dx_template(path, font_path)
        else:
            _render_template(path, text, font_path)
        written.append(filename)

    print(f"font: {font_path}")
    print(f"written: {len(written)}")
    for filename in written:
        print(f"  {filename}")
    print(f"skipped: {len(skipped)}")
    return 0


def _find_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("no suitable font found")


def _render_template(path: Path, text: str, font_path: Path) -> None:
    canvas_size = 160
    max_width = 126
    max_height = 126
    font_size = 112
    while font_size >= 20:
        font = ImageFont.truetype(str(font_path), font_size)
        probe = Image.new("L", (canvas_size, canvas_size), 255)
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            break
        font_size -= 4

    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas_size - width) // 2 - bbox[0]
    y = (canvas_size - height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)
    image.save(path)


def _render_d_over_dx_template(path: Path, font_path: Path) -> None:
    canvas_size = 160
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    top_font = ImageFont.truetype(str(font_path), 62)
    bottom_font = ImageFont.truetype(str(font_path), 56)
    _draw_centered_text(draw, "d", top_font, y_center=45, canvas_size=canvas_size)
    draw.line((42, 80, 118, 80), fill=0, width=5)
    _draw_centered_text(draw, "dx", bottom_font, y_center=118, canvas_size=canvas_size)
    image.save(path)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y_center: int,
    canvas_size: int,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas_size - width) // 2 - bbox[0]
    y = y_center - height // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=0)


if __name__ == "__main__":
    raise SystemExit(main())
