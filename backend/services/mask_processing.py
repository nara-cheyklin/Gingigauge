"""Gingiva mask splitting and tooth-to-gingiva boundary pairing."""

import cv2
import numpy as np

_X_TOL = 5


def split_upper_lower(mask_255: np.ndarray):
    """Return (upper_mask, lower_mask) as uint8 arrays from a binary gingiva mask."""
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


def build_pairs(points: list, upper_mask: np.ndarray, lower_mask: np.ndarray) -> list:
    """Match each detected tooth point to its inner/outer gingiva boundary pixels."""
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
