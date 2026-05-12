"""Image preprocessing utilities (CLAHE, encoding)."""

import base64
import cv2
import numpy as np

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)


def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def ndarray_to_jpeg_bytes(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG.")
    return buf.tobytes()


def ndarray_to_base64_jpeg(image_rgb: np.ndarray) -> str:
    return base64.b64encode(ndarray_to_jpeg_bytes(image_rgb)).decode("utf-8")
