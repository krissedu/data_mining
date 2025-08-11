from typing import Tuple
from PIL import Image
import numpy as np


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def crop_xyxy(img: Image.Image, box_xyxy: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))
