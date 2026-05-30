from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DeskewResult:
    image: np.ndarray
    angle: float


def estimate_skew_angle(binary_image: np.ndarray) -> float:
    if binary_image.ndim != 2:
        raise ValueError("estimate_skew_angle expects a single-channel image")
    coords = np.column_stack(np.where(binary_image < 255))
    if coords.shape[0] < 10:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = float(rect[-1])
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90
    return angle


def rotate_image(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border_value)


def deskew_image(binary_image: np.ndarray, max_correction_angle: float = 20.0) -> DeskewResult:
    angle = estimate_skew_angle(binary_image)
    if abs(angle) > max_correction_angle:
        return DeskewResult(image=binary_image, angle=0.0)
    corrected = rotate_image(binary_image, angle)
    _, corrected = cv2.threshold(corrected, 128, 255, cv2.THRESH_BINARY)
    return DeskewResult(image=corrected, angle=angle)
