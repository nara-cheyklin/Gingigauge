"""FDI tooth-ID assignment for front, right, and left dental views."""

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


def assign_tooth_ids(pairs: list, view: str, image_shape: tuple) -> list:
    """Attach inferred_tooth_id (FDI) and inferred_tooth_id_raw to each pair dict."""
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
