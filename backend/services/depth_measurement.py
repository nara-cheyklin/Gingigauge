"""Depth-based 3-D distance measurement for KGW calculation."""

import numpy as np

_DEPTH_SEARCH_RADIUS = 2


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
    """Return 3-D point, falling back to nearest valid neighbour within search radius."""
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


def measure_pairs(pairs: list, depth: np.ndarray, intr: dict) -> list:
    """Compute tooth_to_inner, tooth_to_outer, and inner_to_outer distances for each pair."""
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
