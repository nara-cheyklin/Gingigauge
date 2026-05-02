"""
Full KGW pipeline service.

Flow:
  image_bytes + depth_map
    -> CLAHE enhancement
    -> GCP Vertex AI segmentation (binary mask)
    -> Roboflow tooth detection
    -> upper/lower mask split
    -> per-tooth inner/outer boundary
    -> depth-based mm measurement
    -> annotated image + JSON output
"""

import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

from backend.config.settings import (
    CAMERA_INTRINSICS,
    DEPTH_UNIT_SCALE,
    KGW_THRESHOLD_MM,
    ROBOFLOW_API_KEY,
)
from backend.services.inference import (
    predict_mask_with_endpoint,
    resize_mask_to_original,
    confidence_logic,
)

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)
_X_TOL = 5
_DEPTH_SEARCH_RADIUS = 2


# --- image helpers ------------------------------------------------------------

def _apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def _ndarray_to_jpeg_bytes(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG.")
    return buf.tobytes()


def _ndarray_to_base64_jpeg(image_rgb: np.ndarray) -> str:
    return base64.b64encode(_ndarray_to_jpeg_bytes(image_rgb)).decode("utf-8")


# --- tooth detection (Roboflow) -----------------------------------------------

def _detect_teeth(image_rgb: np.ndarray) -> list:
    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY is not set.")

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to encode image for Roboflow.")

    resp = requests.post(
        "https://serverless.roboflow.com/toothcariesdetection/2",
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": ("image.jpg", encoded.tobytes(), "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()

    return [
        {
            "cx": int(round(p["x"])),
            "cy": int(round(p["y"])),
            "class": p["class"],
            "confidence": float(p["confidence"]),
        }
        for p in resp.json().get("predictions", [])
    ]


# --- mask processing ----------------------------------------------------------

def _split_upper_lower(mask_255: np.ndarray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask_255 > 0).astype(np.uint8), connectivity=8
    )
    components = []
    for lid in range(1, num_labels):
        if stats[lid, cv2.CC_STAT_AREA] < 20:
            continue
        comp = np.zeros_like(mask_255, dtype=np.uint8)
        comp[labels == lid] = 255
        components.append((centroids[lid][1], comp))

    if len(components) < 2:
        raise ValueError("Could not find two separate gingiva regions in the mask.")

    components.sort(key=lambda x: x[0])
    return components[0][1], components[1][1]


def _column_boundary(mask: np.ndarray, cx: int, x_tol: int = _X_TOL):
    h, w = mask.shape
    best, best_dx = None, float("inf")
    for x in range(max(0, cx - x_tol), min(w, cx + x_tol + 1)):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        dx = abs(x - cx)
        if dx < best_dx:
            best_dx = dx
            best = (x, int(ys.min()), int(ys.max()))
    return best


def _build_pairs(points: list, upper_mask: np.ndarray, lower_mask: np.ndarray) -> list:
    if not points:
        return []

    median_y = sorted(p["cy"] for p in points)[len(points) // 2]
    pairs = []

    for pt in points:
        is_upper = pt["cy"] < median_y
        hit = _column_boundary(upper_mask if is_upper else lower_mask, pt["cx"])
        if hit is None:
            continue
        x, top_y, bot_y = hit
        pairs.append({
            "group": "upper" if is_upper else "lower",
            "tooth_point": (pt["cx"], pt["cy"]),
            "inner_point": (x, bot_y if is_upper else top_y),
            "outer_point": (x, top_y if is_upper else bot_y),
            "class": pt["class"],
            "confidence": pt["confidence"],
        })

    return pairs


# --- depth measurement --------------------------------------------------------

def _pixel_to_3d(u: int, v: int, depth: np.ndarray, intr: dict):
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    z_raw = float(depth[v, u])
    if z_raw <= 0:
        return None
    Z = z_raw * intr["depth_scale"]
    return np.array([
        (u - intr["ppx"]) * Z / intr["fx"],
        (v - intr["ppy"]) * Z / intr["fy"],
        Z,
    ], dtype=np.float64)


def _robust_3d(u: int, v: int, depth: np.ndarray, intr: dict):
    p = _pixel_to_3d(u, v, depth, intr)
    if p is not None:
        return p
    h, w = depth.shape
    best, best_d = None, float("inf")
    r = _DEPTH_SEARCH_RADIUS
    for yy in range(max(0, v - r), min(h, v + r + 1)):
        for xx in range(max(0, u - r), min(w, u + r + 1)):
            candidate = _pixel_to_3d(xx, yy, depth, intr)
            if candidate is None:
                continue
            d = abs(xx - u) + abs(yy - v)
            if d < best_d:
                best_d, best = d, candidate
    return best


def _dist_mm(p1: tuple, p2: tuple, depth: np.ndarray, intr: dict):
    P1 = _robust_3d(p1[0], p1[1], depth, intr)
    P2 = _robust_3d(p2[0], p2[1], depth, intr)
    if P1 is None or P2 is None:
        return None
    return float(np.linalg.norm(P1 - P2) * 1000.0)


def _measure_pairs(pairs: list, depth: np.ndarray, intr: dict) -> list:
    measured = []
    for idx, pair in enumerate(pairs):
        kgw = _dist_mm(pair["inner_point"], pair["outer_point"], depth, intr)
        measured.append({
            "index": idx,
            "group": pair["group"],
            "class": pair["class"],
            "confidence": round(pair["confidence"], 3),
            "tooth_point": list(pair["tooth_point"]),
            "inner_point": list(pair["inner_point"]),
            "outer_point": list(pair["outer_point"]),
            "kgw_mm": round(kgw, 2) if kgw is not None else None,
        })
    return measured


# --- annotation ---------------------------------------------------------------

def _annotate(image_rgb: np.ndarray, measured: list) -> np.ndarray:
    canvas = image_rgb.copy()
    for pair in measured:
        ix, iy = pair["inner_point"]
        ox, oy = pair["outer_point"]
        cv2.line(canvas, (ix, iy), (ox, oy), (0, 200, 255), 2)
        cv2.circle(canvas, (ix, iy), 4, (0, 255, 0), -1)
        cv2.circle(canvas, (ox, oy), 4, (0, 255, 0), -1)
        if pair["kgw_mm"] is not None:
            cv2.putText(
                canvas, f"{pair['kgw_mm']:.1f}mm",
                (ox + 4, oy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return canvas


# --- public entry point -------------------------------------------------------

def run_full_pipeline(image_bytes: bytes, depth_map: np.ndarray,
                      depth_intrinsics: dict = None) -> dict:
    """
    Run the full KGW measurement pipeline.

    Args:
        image_bytes:       Raw bytes of the RGB image (JPEG/PNG).
        depth_map:         2-D numpy array of depth values aligned to the RGB frame.
        depth_intrinsics:  Optional dict with keys fx, fy, ppx, ppy, depth_scale.
                           Falls back to CAMERA_INTRINSICS from settings when omitted.

    Returns:
        {
            "kgw_mm":         float | None   (minimum KGW across detected teeth)
            "confidence":     float           (mean mask confidence 0-1)
            "interpretation": str
            "image_base64":   str             (JPEG annotated image, base64)
            "teeth":          list            (per-tooth measurement detail)
        }
    """
    intr = depth_intrinsics or {
        "fx": CAMERA_INTRINSICS["fx"],
        "fy": CAMERA_INTRINSICS["fy"],
        "ppx": CAMERA_INTRINSICS["cx"],
        "ppy": CAMERA_INTRINSICS["cy"],
        "depth_scale": DEPTH_UNIT_SCALE,
    }

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)
    original_size = image.size  # (width, height)

    # CLAHE then GCP segmentation
    clahe_rgb = _apply_clahe(image_rgb)
    clahe_bytes = _ndarray_to_jpeg_bytes(clahe_rgb)
    mask_small = predict_mask_with_endpoint(clahe_bytes)
    binary_mask = resize_mask_to_original(mask_small, original_size)
    confidence = confidence_logic(binary_mask)

    # tooth detection
    points = _detect_teeth(clahe_rgb)

    # per-tooth KGW measurement
    measured = []
    kgw_mm = None

    if len(points) >= 2:
        upper_mask, lower_mask = _split_upper_lower(binary_mask * 255)
        pairs = _build_pairs(points, upper_mask, lower_mask)
        measured = _measure_pairs(pairs, depth_map, intr)
        valid = [m["kgw_mm"] for m in measured if m["kgw_mm"] is not None]
        kgw_mm = round(min(valid), 2) if valid else None

    if kgw_mm is None:
        interpretation = "Insufficient tooth detection for measurement"
    elif kgw_mm >= KGW_THRESHOLD_MM:
        interpretation = "Adequate keratinized gingiva width"
    else:
        interpretation = "Inadequate keratinized gingiva width"

    return {
        "kgw_mm": kgw_mm,
        "confidence": confidence,
        "interpretation": interpretation,
        "image_base64": _ndarray_to_base64_jpeg(_annotate(image_rgb, measured)),
        "teeth": measured,
    }
