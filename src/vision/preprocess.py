from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    original: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    method: str
    debug_images: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class PreprocessOptions:
    threshold_method: str = "fixed"
    threshold: int = 128
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    median_kernel: int = 3
    gaussian_kernel: int = 0
    use_clahe: bool = False
    normalize_brightness: bool = False
    morph_open: int = 0
    morph_close: int = 0


def read_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    return image


def preprocess_image(
    image_or_path: str | Path | np.ndarray,
    threshold: int = 128,
    median_kernel: int = 3,
    use_otsu: bool = False,
    options: PreprocessOptions | None = None,
) -> PreprocessResult:
    if options is None:
        options = PreprocessOptions(
            threshold_method="otsu" if use_otsu else "fixed",
            threshold=threshold,
            median_kernel=median_kernel,
        )

    if isinstance(image_or_path, (str, Path)):
        original = read_image(image_or_path)
    else:
        original = image_or_path.copy()

    if original.ndim == 2:
        gray = original
    else:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    debug_images: dict[str, np.ndarray] = {"gray": gray}
    working = gray

    if options.normalize_brightness:
        working = cv2.normalize(working, None, 0, 255, cv2.NORM_MINMAX)
        debug_images["brightness_normalized"] = working

    if options.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        working = clahe.apply(working)
        debug_images["clahe"] = working

    if options.gaussian_kernel > 1:
        if options.gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be odd")
        working = cv2.GaussianBlur(working, (options.gaussian_kernel, options.gaussian_kernel), 0)
        debug_images["gaussian"] = working

    if options.threshold_method == "otsu":
        _, binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method = "otsu"
    elif options.threshold_method == "adaptive":
        block_size = options.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            working,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(3, block_size),
            options.adaptive_c,
        )
        method = f"adaptive_{block_size}_{options.adaptive_c}"
    elif options.threshold_method == "fixed":
        _, binary = cv2.threshold(working, options.threshold, 255, cv2.THRESH_BINARY)
        method = f"fixed_{options.threshold}"
    else:
        raise ValueError(f"unknown threshold method: {options.threshold_method}")

    if options.median_kernel > 1:
        if options.median_kernel % 2 == 0:
            raise ValueError("median_kernel must be odd")
        binary = cv2.medianBlur(binary, options.median_kernel)
        debug_images["median"] = binary

    if options.morph_open > 1:
        kernel = np.ones((options.morph_open, options.morph_open), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        debug_images["morph_open"] = binary

    if options.morph_close > 1:
        kernel = np.ones((options.morph_close, options.morph_close), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        debug_images["morph_close"] = binary

    debug_images["binary"] = binary

    return PreprocessResult(original=original, gray=gray, binary=binary, method=method, debug_images=debug_images)


def save_debug_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"failed to write debug image: {output}")
    encoded.tofile(str(output))
    return output
