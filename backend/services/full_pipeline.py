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

import base64
import logging
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

from backend.config.settings import (
    RECESSION_THRESHOLD,
    RECESSION_CONCERN_THRESHOLD,
    ROBOFLOW_API_KEY,
)
from backend.services.inference import (
    predict_with_endpoint,
    resize_mask_to_original,
)

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = (8, 8)
_X_TOL = 5
_DEPTH_SEARCH_RADIUS = 2

# --- FDI view config ----------------------------------------------------------

_FRONT_ALLOWED = {
    "upper": {"18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28"},
    "lower": {"48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"},
}
_RIGHT_ALLOWED = {
    "upper": {"18","17","16","15","14","13","12","11"},
    "lower": {"48","47","46","45","44","43","42","41"},
}
_LEFT_ALLOWED = {
    "upper": {"21","22","23","24","25","26","27","28"},
    "lower": {"31","32","33","34","35","36","37","38"},
}


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
            "class": p.get("class", "Tooth"),
            "confidence": float(p["confidence"]),
            "det_width": float(p.get("width", 0.0)),
            "det_height": float(p.get("height", 0.0)),
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


def _split_points_upper_lower(points: list):
    if not points:
        return [], []
    ys = sorted(p["cy"] for p in points)
    median_y = ys[len(ys) // 2]
    return [p for p in points if p["cy"] < median_y], [p for p in points if p["cy"] >= median_y]


def _build_pairs(points: list, upper_mask: np.ndarray, lower_mask: np.ndarray) -> list:
    upper_pts, lower_pts = _split_points_upper_lower(points)
    pairs = []

    for group, pts, mask in (("upper", upper_pts, upper_mask), ("lower", lower_pts, lower_mask)):
        for pt in pts:
            hit = _column_boundary(mask, pt["cx"])
            if hit is None:
                continue
            x, top_y, bot_y = hit
            pairs.append({
                "group": group,
                "tooth_point": (pt["cx"], pt["cy"]),
                "inner_point": (x, bot_y if group == "upper" else top_y),
                "outer_point": (x, top_y if group == "upper" else bot_y),
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
    Z = z_raw * intr["depth_scale"]  # mm
    if Z <= 0:
        return None
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
    return float(np.linalg.norm(P1 - P2))


def _measure_pairs(pairs: list, depth: np.ndarray, intr: dict) -> list:
    measured = []
    for idx, pair in enumerate(pairs):
        tooth_to_inner = _dist_mm(pair["tooth_point"], pair["inner_point"], depth, intr)
        tooth_to_outer = _dist_mm(pair["tooth_point"], pair["outer_point"], depth, intr)
        inner_to_outer = _dist_mm(pair["inner_point"], pair["outer_point"], depth, intr)
        measured.append({
            "index": idx,
            "group": pair["group"],
            "class": pair["class"],
            "confidence": round(pair["confidence"], 3),
            "tooth_point": list(pair["tooth_point"]),
            "inner_point": list(pair["inner_point"]),
            "outer_point": list(pair["outer_point"]),
            "tooth_to_inner_mm": round(tooth_to_inner, 2) if tooth_to_inner is not None else None,
            "tooth_to_outer_mm": round(tooth_to_outer, 2) if tooth_to_outer is not None else None,
            "inner_to_outer_mm": round(inner_to_outer, 2) if inner_to_outer is not None else None,
            "kgw_mm": round(inner_to_outer, 2) if inner_to_outer is not None else None,
        })
    return measured


# --- FDI tooth mapping --------------------------------------------------------

def _map_front_view(pairs: list, image_shape: tuple) -> dict:
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    groups: dict = {"ul": [], "ur": [], "ll": [], "lr": []}
    for p in pairs:
        x, y = p["tooth_point"]
        key = ("u" if y < cy else "l") + ("l" if x < cx else "r")
        groups[key].append(p)

    def sort_key(p):
        return (abs(p["tooth_point"][0] - cx), abs(p["tooth_point"][1] - cy))

    for g in groups.values():
        g.sort(key=sort_key)

    seqs = {
        "ul": ["11","12","13","14","15","16","17","18"],
        "ur": ["21","22","23","24","25","26","27","28"],
        "ll": ["41","42","43","44","45","46","47","48"],
        "lr": ["31","32","33","34","35","36","37","38"],
    }
    mapping = {}
    for key, seq in seqs.items():
        for i, p in enumerate(groups[key]):
            if i < len(seq):
                mapping[p["index"]] = seq[i]
    return mapping


def _center_align_sequence(sorted_pairs: list, sequence: list) -> dict:
    n, m = len(sorted_pairs), len(sequence)
    if n <= 0:
        return {}
    if n >= m:
        return {p["index"]: t for p, t in zip(sorted_pairs[:m], sequence)}
    offset = max(0, (m - n) // 2)
    return {p["index"]: t for p, t in zip(sorted_pairs, sequence[offset:offset + n])}


def _map_side_view(pairs: list, view: str) -> dict:
    mapping = {}
    for group in ("upper", "lower"):
        gps = sorted([p for p in pairs if p["group"] == group], key=lambda p: p["tooth_point"][0])
        if view == "right":
            seq = (["18","17","16","15","14","13","12","11"] if group == "upper"
                   else ["48","47","46","45","44","43","42","41"])
        else:
            seq = (["21","22","23","24","25","26","27","28"] if group == "upper"
                   else ["31","32","33","34","35","36","37","38"])
        mapping.update(_center_align_sequence(gps, seq))
    return mapping


def _filter_by_view(pairs: list, mapping: dict, view: str) -> dict:
    allowed = {"front": _FRONT_ALLOWED, "right": _RIGHT_ALLOWED, "left": _LEFT_ALLOWED}.get(view)
    if allowed is None:
        return mapping
    return {
        p["index"]: mapping[p["index"]]
        for p in pairs
        if p["index"] in mapping and mapping[p["index"]] in allowed.get(p["group"], set())
    }


def _assign_tooth_ids(pairs: list, view: str, image_shape: tuple) -> list:
    if view in ("front", "unknown", None):
        raw_map = _map_front_view(pairs, image_shape)
    else:
        raw_map = _map_side_view(pairs, view)
    filtered_map = _filter_by_view(pairs, raw_map, view or "front")
    return [
        dict(p,
             inferred_tooth_id=filtered_map.get(p["index"]),
             inferred_tooth_id_raw=raw_map.get(p["index"]))
        for p in pairs
    ]


# --- annotation ---------------------------------------------------------------

def _annotate(image_rgb: np.ndarray, measured: list) -> np.ndarray:
    canvas = image_rgb.copy()
    for pair in measured:
        ix, iy = pair["inner_point"]
        ox, oy = pair["outer_point"]
        tooth_id = pair.get("inferred_tooth_id")

        cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)    # red: inner border
        cv2.circle(canvas, (ox, oy), 4, (255, 255, 0), -1)  # yellow: outer border
        cv2.line(canvas, (ix, iy), (ox, oy), (0, 255, 0), 2)

        label = str(tooth_id) if tooth_id else str(pair["index"])
        cv2.putText(canvas, label, (ix + 5, iy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --- public entry point -------------------------------------------------------

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
    intr = dict(depth_intrinsics)  # local copy — depth_scale may be adjusted below

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)
    original_size = image.size  # (width, height) PIL convention

    # --- depth_scale auto-correction ----------------------------------------
    depth_scale = intr.get("depth_scale", 1.0)
    sample_centre = depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2]
    z_centre_mm = float(sample_centre) * depth_scale
    logger.info("depth_scale=%.6f  depth centre raw=%.1f  Z=%.1f mm",
                depth_scale, float(sample_centre), z_centre_mm)

    if z_centre_mm > 500.0 and float(sample_centre) > 0:
        # Bag depth likely in 0.1 mm/unit (raw ~3000 for 300 mm) — shrink by 10×
        candidate = depth_scale * 0.1
        if 50.0 <= float(sample_centre) * candidate <= 500.0:
            intr["depth_scale"] = candidate
            depth_scale = candidate
            z_centre_mm = float(sample_centre) * depth_scale
            logger.warning("depth_scale auto-corrected to %.6f  (centre Z now %.1f mm)",
                           depth_scale, z_centre_mm)
    elif 0 < z_centre_mm < 50.0:
        candidate = depth_scale * 10.0
        if 50.0 <= float(sample_centre) * candidate <= 500.0:
            intr["depth_scale"] = candidate
            depth_scale = candidate
            z_centre_mm = float(sample_centre) * depth_scale
            logger.warning("depth_scale auto-corrected to %.6f  (centre Z now %.1f mm)",
                           depth_scale, z_centre_mm)

    if not (50.0 <= z_centre_mm <= 500.0):
        logger.warning("Centre depth Z=%.1f mm is outside 50–500 mm; "
                       "depth_scale (%.6f) or encoding may still be wrong.",
                       z_centre_mm, depth_scale)

    # --- guard: depth and image must be the same resolution -----------------
    img_h, img_w = image_rgb.shape[:2]
    dep_h, dep_w = depth_map.shape[:2]
    if img_h != dep_h or img_w != dep_w:
        raise ValueError(
            f"depth_map ({dep_w}×{dep_h}) and image ({img_w}×{img_h}) "
            "must be the same resolution. "
            "Make sure the bag contains aligned depth "
            "(/camera/aligned_depth_to_color/image_raw)."
        )

    # CLAHE then GCP segmentation
    clahe_rgb = _apply_clahe(image_rgb)
    clahe_bytes = _ndarray_to_jpeg_bytes(clahe_rgb)
    segmentation = predict_with_endpoint(clahe_bytes)
    mask_small = segmentation["mask"]
    binary_mask = resize_mask_to_original(mask_small, original_size)

    # tooth detection
    points = _detect_teeth(clahe_rgb)

    # per-tooth KGW measurement
    measured = []
    kgw_mm = None

    if len(points) >= 2:
        upper_mask, lower_mask = _split_upper_lower(binary_mask * 255)
        pairs = _build_pairs(points, upper_mask, lower_mask)
        measured = _measure_pairs(pairs, depth_map, intr)
        measured = _assign_tooth_ids(measured, view, image_rgb.shape)
        valid = [m["kgw_mm"] for m in measured if m["kgw_mm"] is not None]
        kgw_mm = round(min(valid), 2) if valid else None

    if kgw_mm is None:
        interpretation = "Insufficient tooth detection for measurement"
    elif kgw_mm < RECESSION_THRESHOLD:
        interpretation = "Recession"
    elif kgw_mm>= RECESSION_THRESHOLD and kgw_mm < RECESSION_CONCERN_THRESHOLD:
        interpretation = "At Risk"
    else:
        interpretation = "Healthy"

    return {
        "kgw_mm": kgw_mm,
        "interpretation": interpretation,
        "image_base64": _ndarray_to_base64_jpeg(_annotate(image_rgb, measured)),
        "teeth": measured,
        "view": view,
    }
