from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def pil_to_rgb_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def rgb_numpy_to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")


def gray_numpy_to_pil(image_gray: np.ndarray) -> Image.Image:
    return Image.fromarray(image_gray.astype(np.uint8), mode="L")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
