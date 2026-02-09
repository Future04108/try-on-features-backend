from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def load_image_rgba(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pil_to_base64_png(img: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(img)).decode("utf-8")


def base64_png_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr


def numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def clamp_size_keep_aspect(w: int, h: int, max_side: int) -> Tuple[int, int]:
    if max(w, h) <= max_side:
        return w, h
    if w >= h:
        nw = max_side
        nh = int(round(h * (max_side / w)))
    else:
        nh = max_side
        nw = int(round(w * (max_side / h)))
    return max(1, nw), max(1, nh)


