from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_ROOT = PROJECT_ROOT / "data" / "samples"
DEFAULT_LABEL_ROOT = PROJECT_ROOT / "data" / "labels"

FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
)


@dataclass(frozen=True)
class SampleSpec:
    category: str
    stem: str
    label: str
    renderer: str = "row"
    rotate_degrees: float = 0.0
    blur_radius: float = 0.0


SAMPLES: tuple[SampleSpec, ...] = (
    SampleSpec("printed_basic", "printed_basic_021", "23+45=68"),
    SampleSpec("printed_basic", "printed_basic_022", "72-18=54", rotate_degrees=-1.0),
    SampleSpec("printed_basic", "printed_basic_023", "11x6=66"),
    SampleSpec("printed_basic", "printed_basic_024", "(4+5)x7=63", rotate_degrees=1.0),
    SampleSpec("printed_basic", "printed_basic_025", "96÷12=8"),
    SampleSpec("printed_basic", "printed_basic_026", "34+27=61"),
    SampleSpec("printed_basic", "printed_basic_027", "85-49=36", rotate_degrees=-0.8),
    SampleSpec("printed_basic", "printed_basic_028", "13x4=52"),
    SampleSpec("printed_basic", "printed_basic_029", "144÷12=12"),
    SampleSpec("printed_basic", "printed_basic_030", "(6+2)x5=40"),
    SampleSpec("printed_basic", "printed_basic_031", "90÷10=9", rotate_degrees=0.8),
    SampleSpec("printed_basic", "printed_basic_032", "17+29=46"),
    SampleSpec("printed_basic", "printed_basic_033", "64-28=36"),
    SampleSpec("printed_basic", "printed_basic_034", "8x9=72"),
    SampleSpec("printed_basic", "printed_basic_035", "(9-4)x6=30"),
    SampleSpec("printed_basic", "printed_basic_036", "15+16=31"),
    SampleSpec("printed_basic", "printed_basic_037", "42÷7=6"),
    SampleSpec("printed_basic", "printed_basic_038", "21x3=63"),
    SampleSpec("printed_basic", "printed_basic_039", "100-37=63"),
    SampleSpec("printed_basic", "printed_basic_040", "(7+8)÷3=5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_021", "-12+5=-7"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_022", "4.5+1.5=6"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_023", "8.75-3.25=5.5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_024", "(-3)x(-4)=12"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_025", "-10÷2=-5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_026", "-14+9=-5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_027", "2.25+3.75=6"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_028", "9.6÷3.2=3"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_029", "(-5)+8=3"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_030", "-4x6=-24"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_031", "12.5-7.5=5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_032", "0.3+0.7=1"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_033", "5-8=-3"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_034", "(-6)÷3=-2"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_035", "1.2x5=6"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_036", "-0.5+1.5=1"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_037", "7.25-2.25=5"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_038", "(-2)x5=-10"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_039", "18÷(-3)=-6"),
    SampleSpec("printed_decimal_negative", "printed_decimal_negative_040", "-7.5+2.5=-5"),
    SampleSpec("printed_2d_layout", "printed_2d_021", "2/5", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_022", "(y+2)/3", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_023", "y^3", renderer="superscript"),
    SampleSpec("printed_2d_layout", "printed_2d_024", "y_2+1", renderer="subscript_row"),
    SampleSpec("printed_2d_layout", "printed_2d_025", "4/7", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_026", "3/8", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_027", "(y+4)/5", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_028", "y^4", renderer="superscript"),
    SampleSpec("printed_2d_layout", "printed_2d_029", "y_3+2", renderer="subscript_row"),
    SampleSpec("printed_2d_layout", "printed_2d_030", "6/11", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_031", "(2+3)/4", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_032", "y^5", renderer="superscript"),
    SampleSpec("printed_2d_layout", "printed_2d_033", "7/13", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_034", "(y+1)/2", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_035", "8/9", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_036", "5/12", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_037", "y^6", renderer="superscript"),
    SampleSpec("printed_2d_layout", "printed_2d_038", "y_5+3", renderer="subscript_row"),
    SampleSpec("printed_2d_layout", "printed_2d_039", "(3+4)/7", renderer="fraction"),
    SampleSpec("printed_2d_layout", "printed_2d_040", "9/10", renderer="fraction"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic sample images and matching labels.")
    parser.add_argument("--sample-root", default=str(DEFAULT_SAMPLE_ROOT))
    parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    parser.add_argument("--category", choices=sorted({sample.category for sample in SAMPLES}))
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated samples.")
    parser.add_argument("--dry-run", action="store_true", help="List generated paths without writing files.")
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    label_root = Path(args.label_root)
    font_path = _find_font()
    specs = [sample for sample in SAMPLES if args.category is None or sample.category == args.category]

    written = 0
    skipped = 0
    for spec in specs:
        image_path = sample_root / spec.category / f"{spec.stem}.png"
        label_path = label_root / spec.category / f"{spec.stem}.txt"
        if (image_path.exists() or label_path.exists()) and not args.force:
            skipped += 1
            print(f"skip existing: {image_path}")
            continue
        print(f"write: {image_path} [{spec.label}]")
        if args.dry_run:
            continue
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image = _render_sample(spec, font_path)
        image.save(image_path)
        label_path.write_text(spec.label, encoding="utf-8")
        written += 1

    print(f"font: {font_path}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    return 0


def _find_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("no suitable font found")


def _render_sample(spec: SampleSpec, font_path: Path) -> Image.Image:
    if spec.renderer == "row":
        image = _render_row(spec.label, font_path)
    elif spec.renderer == "fraction":
        numerator, denominator = spec.label.split("/", 1)
        image = _render_fraction(numerator, denominator, font_path)
    elif spec.renderer == "superscript":
        image = _render_superscript(spec.label, font_path)
    elif spec.renderer == "subscript_row":
        image = _render_subscript_row(spec.label, font_path)
    elif spec.renderer == "sqrt":
        inside = spec.label.removeprefix("sqrt(").removesuffix(")")
        image = _render_sqrt(inside, font_path)
    else:
        raise ValueError(f"unknown renderer: {spec.renderer}")

    if spec.rotate_degrees:
        image = image.rotate(spec.rotate_degrees, expand=True, fillcolor=255)
        image = _pad_to_canvas(_crop_foreground(image), min_width=220, min_height=96, padding=24)
    if spec.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(spec.blur_radius))
    return image


def _render_row(text: str, font_path: Path) -> Image.Image:
    font = _fit_font(text, font_path, max_width=440, max_height=72, start_size=64)
    probe = Image.new("L", (1, 1), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = Image.new("L", (width + 48, height + 48), 255)
    draw = ImageDraw.Draw(image)
    draw.text((24 - bbox[0], 24 - bbox[1]), text, fill=0, font=font)
    return _pad_to_canvas(_crop_foreground(image), min_width=220, min_height=96, padding=24)


def _render_fraction(numerator: str, denominator: str, font_path: Path) -> Image.Image:
    top_font = _fit_font(numerator, font_path, max_width=170, max_height=48, start_size=48)
    bottom_font = _fit_font(denominator, font_path, max_width=170, max_height=48, start_size=48)
    image = Image.new("L", (240, 150), 255)
    draw = ImageDraw.Draw(image)
    _draw_centered(draw, numerator, top_font, center_x=120, center_y=42)
    _draw_centered(draw, denominator, bottom_font, center_x=120, center_y=112)
    draw.line((54, 76, 186, 76), fill=0, width=4)
    return _pad_to_canvas(_crop_foreground(image), min_width=180, min_height=120, padding=20)


def _render_superscript(text: str, font_path: Path) -> Image.Image:
    base, exponent = text.split("^", 1)
    base_font = ImageFont.truetype(str(font_path), 58)
    script_font = ImageFont.truetype(str(font_path), 34)
    image = Image.new("L", (170, 110), 255)
    draw = ImageDraw.Draw(image)
    draw.text((44, 38), base, fill=0, font=base_font)
    draw.text((88, 16), exponent, fill=0, font=script_font)
    return _pad_to_canvas(_crop_foreground(image), min_width=130, min_height=88, padding=20)


def _render_subscript_row(text: str, font_path: Path) -> Image.Image:
    base, rest = text.split("_", 1)
    subscript, suffix = rest[0], rest[1:]
    base_font = ImageFont.truetype(str(font_path), 58)
    script_font = ImageFont.truetype(str(font_path), 32)
    suffix_font = ImageFont.truetype(str(font_path), 52)
    image = Image.new("L", (250, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.text((30, 36), base, fill=0, font=base_font)
    draw.text((78, 70), subscript, fill=0, font=script_font)
    draw.text((112, 42), suffix, fill=0, font=suffix_font)
    return _pad_to_canvas(_crop_foreground(image), min_width=180, min_height=96, padding=20)


def _render_sqrt(inside: str, font_path: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_path), 56)
    image = Image.new("L", (230, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.line((26, 72, 44, 94), fill=0, width=4)
    draw.line((44, 94, 66, 30), fill=0, width=4)
    draw.line((66, 30, 178, 30), fill=0, width=4)
    draw.text((76, 38), inside, fill=0, font=font)
    return _pad_to_canvas(_crop_foreground(image), min_width=170, min_height=100, padding=20)


def _fit_font(text: str, font_path: Path, max_width: int, max_height: int, start_size: int) -> ImageFont.FreeTypeFont:
    probe = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(probe)
    for size in range(start_size, 15, -2):
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
    return ImageFont.truetype(str(font_path), 16)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: int,
    center_y: int,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center_x - width // 2 - bbox[0], center_y - height // 2 - bbox[1]), text, fill=0, font=font)


def _crop_foreground(image: Image.Image) -> Image.Image:
    bbox = Image.eval(image, lambda px: 255 - px).getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def _pad_to_canvas(image: Image.Image, min_width: int, min_height: int, padding: int) -> Image.Image:
    width = max(min_width, image.width + padding * 2)
    height = max(min_height, image.height + padding * 2)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
