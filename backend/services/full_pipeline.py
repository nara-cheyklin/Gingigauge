"""
Full KGW pipeline service.

Flow:
  image_bytes + depth_map
    -> CLAHE enhancement
    -> GCP Vertex AI segmentation (binary mask)
    -> Roboflow tooth detection
    -> upper/lower mask + point split
    -> per-tooth inner/outer boundary
    -> depth-based mm measurement (3 distances per tooth)
    -> FDI tooth ID mapping
    -> annotated image + JSON output
"""

import logging

import cv2
import numpy as np
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

from backend.config.settings import (
    RECESSION_THRESHOLD,
    RECESSION_CONCERN_THRESHOLD,
)
from backend.services.inference import predict_with_endpoint, resize_mask_to_original
from backend.services.image_utils import apply_clahe, ndarray_to_jpeg_bytes, ndarray_to_base64_jpeg
from backend.services.tooth_detection import detect_teeth
from backend.services.mask_processing import split_upper_lower, build_pairs
from backend.services.depth_measurement import measure_pairs
from backend.services.fdi_mapping import assign_tooth_ids


def _annotate(image_rgb: np.ndarray, measured: list) -> np.ndarray:
    canvas = image_rgb.copy()
    for pair in measured:
        ix, iy = pair["inner_point"]
        ox, oy = pair["outer_point"]
        tooth_id = pair.get("inferred_tooth_id")

        cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (ox, oy), 4, (255, 255, 0), -1)
        cv2.line(canvas, (ix, iy), (ox, oy), (0, 255, 0), 2)

        label = str(tooth_id) if tooth_id else str(pair["index"])
        cv2.putText(canvas, label, (ix + 5, iy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _correct_depth_scale(intr: dict, depth_map: np.ndarray) -> dict:
    """Return a (possibly adjusted) copy of intr with a sane depth_scale."""
    intr = dict(intr)
    depth_scale = intr.get("depth_scale", 1.0)
    sample_centre = depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2]
    z_centre_mm = float(sample_centre) * depth_scale
    logger.info("depth_scale=%.6f  depth centre raw=%.1f  Z=%.1f mm",
                depth_scale, float(sample_centre), z_centre_mm)

    if z_centre_mm > 500.0 and float(sample_centre) > 0:
        candidate = depth_scale * 0.1
        if 50.0 <= float(sample_centre) * candidate <= 500.0:
            intr["depth_scale"] = candidate
            z_centre_mm = float(sample_centre) * candidate
            logger.warning("depth_scale auto-corrected to %.6f  (centre Z now %.1f mm)",
                           candidate, z_centre_mm)
    elif 0 < z_centre_mm < 50.0:
        candidate = depth_scale * 10.0
        if 50.0 <= float(sample_centre) * candidate <= 500.0:
            intr["depth_scale"] = candidate
            z_centre_mm = float(sample_centre) * candidate
            logger.warning("depth_scale auto-corrected to %.6f  (centre Z now %.1f mm)",
                           candidate, z_centre_mm)

    if not (50.0 <= z_centre_mm <= 500.0):
        logger.warning("Centre depth Z=%.1f mm is outside 50–500 mm; "
                       "depth_scale (%.6f) or encoding may still be wrong.",
                       z_centre_mm, intr["depth_scale"])
    return intr


def _interpret(kgw_mm) -> str:
    if kgw_mm is None:
        return "Insufficient tooth detection for measurement"
    if kgw_mm < RECESSION_THRESHOLD:
        return "Recession"
    if kgw_mm < RECESSION_CONCERN_THRESHOLD:
        return "At Risk"
    return "Healthy"


def run_full_pipeline(image_bytes: bytes, depth_map: np.ndarray,
                      depth_intrinsics: dict, view: str = "front") -> dict:
    """
    Run the full KGW measurement pipeline.

    Args:
        image_bytes:      Raw bytes of the RGB image (JPEG/PNG).
        depth_map:        2-D numpy array of depth values aligned to the RGB frame.
        depth_intrinsics: Dict with keys fx, fy, ppx, ppy, depth_scale.
        view:             Dental view — 'front', 'right', or 'left'.

    Returns:
        {
            "kgw_mm":         float | None   (minimum KGW across detected teeth)
            "interpretation": str
            "image_base64":   str             (JPEG annotated image, base64)
            "teeth":          list            (per-tooth measurement detail)
            "view":           str
        }
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)
    original_size = image.size  # (width, height) — PIL convention

    intr = _correct_depth_scale(depth_intrinsics, depth_map)

    img_h, img_w = image_rgb.shape[:2]
    dep_h, dep_w = depth_map.shape[:2]
    if img_h != dep_h or img_w != dep_w:
        raise ValueError(
            f"depth_map ({dep_w}×{dep_h}) and image ({img_w}×{img_h}) "
            "must be the same resolution. "
            "Make sure the bag contains aligned depth "
            "(/camera/aligned_depth_to_color/image_raw)."
        )

    clahe_rgb = apply_clahe(image_rgb)
    segmentation = predict_with_endpoint(ndarray_to_jpeg_bytes(clahe_rgb))
    binary_mask = resize_mask_to_original(segmentation["mask"], original_size)

    points = detect_teeth(clahe_rgb)

    measured = []
    kgw_mm = None

    if len(points) >= 2:
        upper_mask, lower_mask = split_upper_lower(binary_mask * 255)
        pairs = build_pairs(points, upper_mask, lower_mask)
        measured = measure_pairs(pairs, depth_map, intr)
        measured = assign_tooth_ids(measured, view, image_rgb.shape)
        valid = [m["kgw_mm"] for m in measured if m["kgw_mm"] is not None]
        kgw_mm = round(min(valid), 2) if valid else None

    return {
        "kgw_mm": kgw_mm,
        "interpretation": _interpret(kgw_mm),
        "image_base64": ndarray_to_base64_jpeg(_annotate(image_rgb, measured)),
        "teeth": measured,
        "view": view,
    }
