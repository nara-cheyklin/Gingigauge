import os
import cv2
import json
import argparse
import torch
import numpy as np
import requests
import pyrealsense2 as rs

from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# =========================================================
# ENV
# =========================================================
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "ROBOFLOW_API_KEY not found. Make sure your .env file is set correctly."
    )


# =========================================================
# DEFAULT CONFIG
# =========================================================
DEFAULT_BAG_PATH = r"C:\Users\Beam\Documents\TEST_SUBJECT\NON\NON_FRONT.bag"
DEFAULT_OUTPUT_DIR = r"C:\testing_gingigauge\NON"
DEFAULT_SEGFORMER_MODEL_PATH = "segformer-gingiva-final"

PAUSE_KEY = ord("p")
QUIT_KEY = ord("q")
PLAYBACK_DELAY_MS = 50

X_TOL = 5
DEPTH_SEARCH_RADIUS = 2

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

VALID_MODES = [
    "no_clahe",
    "clahe_detection",
    "clahe_segmentation",
    "clahe_both",
]


# =========================================================
# VIEW / FDI CONFIG
# =========================================================
FRONT_ALLOWED = {
    # Full FDI range for front-view quadrant mapping.
    # The MAE script should use the same tooth set when FRONT_ONLY_EVAL=True.
    "upper": {"18", "17", "16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26", "27", "28"},
    "lower": {"48", "47", "46", "45", "44", "43", "42", "41", "31", "32", "33", "34", "35", "36", "37", "38"},
}

RIGHT_ALLOWED = {
    "upper": {"18", "17", "16", "15", "14", "13", "12", "11"},
    "lower": {"48", "47", "46", "45", "44", "43", "42", "41"},
}

LEFT_ALLOWED = {
    "upper": {"21", "22", "23", "24", "25", "26", "27", "28"},
    "lower": {"31", "32", "33", "34", "35", "36", "37", "38"},
}


# =========================================================
# GENERAL HELPERS
# =========================================================
def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def save_rgb(path: str, image_rgb: np.ndarray) -> None:
    """
    Save RGB image correctly using OpenCV.
    OpenCV imwrite expects BGR, so convert RGB -> BGR.
    """
    cv2.imwrite(path, image_rgb)


def load_rgb(path: str) -> np.ndarray:
    image_rgb = cv2.imread(path)
    if image_rgb is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image_rgb


def infer_view_from_bag_path(bag_path: str) -> str:
    name = Path(bag_path).stem.lower()

    if "front" in name:
        return "front"
    if "right" in name:
        return "right"
    if "left" in name:
        return "left"

    return "unknown"


# =========================================================
# IMAGE ENHANCEMENT
# =========================================================
def apply_clahe_rgb(
    image_rgb: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size=CLAHE_TILE_GRID_SIZE
) -> np.ndarray:
    """
    Apply CLAHE on L-channel in LAB color space.
    Input: RGB image
    Output: RGB image
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return image_clahe


# =========================================================
# REALSENSE HELPERS
# =========================================================
def depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def crop_with_box(image: np.ndarray, box):
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def get_depth_scale(profile):
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()
    except Exception:
        return 0.001


def intrinsics_to_dict(intr, x_offset, y_offset, crop_w, crop_h, depth_scale):
    return {
        "width": int(crop_w),
        "height": int(crop_h),
        "ppx": float(intr.ppx - x_offset),
        "ppy": float(intr.ppy - y_offset),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "depth_scale": float(depth_scale),
    }


def capture_and_crop_from_bag_manual(bag_path: str):
    """
    Original manual workflow:
    1. stream bag
    2. press p to pause
    3. draw ROI manually
    4. return RGB crop, depth crop, depth intrinsics
    """
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    paused_color_rgb = None
    paused_depth_image = None
    paused_depth_vis = None
    paused_intr = None
    paused_depth_scale = None
    frame_index = 0

    print("Streaming bag...")
    print("Press 'p' to pause on current frame.")
    print("Press 'q' to quit.")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                print("End of bag reached.")
                break

            try:
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            except RuntimeError as e:
                print(f"Warning: align failed at frame {frame_index}, skipping. Error: {e}")
                frame_index += 1
                continue

            if not depth_frame or not color_frame:
                frame_index += 1
                continue

            color_image_bgr = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_vis = depth_to_colormap(depth_image)

            display_rgb = color_image_rgb.copy()
            cv2.putText(
                display_rgb,
                f"Frame: {frame_index} | Press 'p' to pause, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Convert for correct display in OpenCV window.
            cv2.imshow("RGB Stream", display_rgb)
            cv2.imshow("Depth Visualization", depth_vis)

            key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF

            if key == PAUSE_KEY:
                paused_color_rgb = color_image_rgb.copy()
                paused_depth_image = depth_image.copy()
                paused_depth_vis = depth_vis.copy()
                paused_intr = depth_frame.profile.as_video_stream_profile().intrinsics
                paused_depth_scale = get_depth_scale(profile)

                print(f"Paused at frame {frame_index}")
                break

            if key == QUIT_KEY:
                raise KeyboardInterrupt("Quit requested by user.")

            frame_index += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if paused_color_rgb is None or paused_depth_image is None or paused_intr is None:
        raise RuntimeError("No frame was paused/captured.")

    print("Draw a rectangle on the RGB image, then press ENTER or SPACE.")
    print("Press C to cancel selection.")

    # cv2.selectROI expects BGR-looking display, so show converted image.
    paused_color_bgr_for_roi = paused_color_rgb

    roi = cv2.selectROI(
        "Select ROI on RGB",
        paused_color_bgr_for_roi,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow("Select ROI on RGB")

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")

    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

    rgb_crop = crop_with_box(paused_color_rgb, roi)
    depth_crop = crop_with_box(paused_depth_image, roi)
    depth_vis_crop = crop_with_box(paused_depth_vis, roi)

    cropped_intr = intrinsics_to_dict(
        paused_intr,
        x_offset=x,
        y_offset=y,
        crop_w=w,
        crop_h=h,
        depth_scale=paused_depth_scale
    )

    return {
        "rgb_crop": rgb_crop,
        "depth_crop": depth_crop,
        "depth_crop_vis": depth_vis_crop,
        "roi": [int(x), int(y), int(w), int(h)],
        "intrinsics": cropped_intr,
        "loaded_from_cache": False,
    }


# Keep old function name for compatibility.
capture_and_crop_from_bag = capture_and_crop_from_bag_manual


# =========================================================
# CROP CACHE
# =========================================================
def crop_cache_exists(cache_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(cache_dir, "rgb_crop.png"))
        and os.path.exists(os.path.join(cache_dir, "depth_crop.npy"))
        and os.path.exists(os.path.join(cache_dir, "intrinsics.json"))
    )


def save_crop_cache(cache_dir: str, bag_data: dict) -> None:
    ensure_dir(cache_dir)

    save_rgb(
        os.path.join(cache_dir, "rgb_crop.png"),
        bag_data["rgb_crop"]
    )

    np.save(
        os.path.join(cache_dir, "depth_crop.npy"),
        bag_data["depth_crop"]
    )

    cv2.imwrite(
        os.path.join(cache_dir, "depth_crop_vis.png"),
        bag_data["depth_crop_vis"]
    )

    save_json(
        os.path.join(cache_dir, "intrinsics.json"),
        bag_data["intrinsics"]
    )

    save_json(
        os.path.join(cache_dir, "roi.json"),
        {
            "roi": bag_data.get("roi"),
            "source": "manual_crop",
        }
    )


def load_crop_cache(cache_dir: str) -> dict:
    rgb_crop = load_rgb(os.path.join(cache_dir, "rgb_crop.png"))
    depth_crop = np.load(os.path.join(cache_dir, "depth_crop.npy"))

    depth_vis_path = os.path.join(cache_dir, "depth_crop_vis.png")

    if os.path.exists(depth_vis_path):
        depth_crop_vis = cv2.imread(depth_vis_path)
        if depth_crop_vis is None:
            depth_crop_vis = depth_to_colormap(depth_crop)
    else:
        depth_crop_vis = depth_to_colormap(depth_crop)

    with open(os.path.join(cache_dir, "intrinsics.json"), "r", encoding="utf-8") as f:
        intr = json.load(f)

    roi = None
    roi_path = os.path.join(cache_dir, "roi.json")

    if os.path.exists(roi_path):
        with open(roi_path, "r", encoding="utf-8") as f:
            roi_data = json.load(f)
            roi = roi_data.get("roi")

    return {
        "rgb_crop": rgb_crop,
        "depth_crop": depth_crop,
        "depth_crop_vis": depth_crop_vis,
        "intrinsics": intr,
        "roi": roi,
        "loaded_from_cache": True,
    }


def get_or_create_manual_crop_cache(bag_path: str, cache_dir: str) -> dict:
    if crop_cache_exists(cache_dir):
        print(f"Using cached crop: {cache_dir}")
        return load_crop_cache(cache_dir)

    print(f"No crop cache found. Creating manual crop for: {bag_path}")
    bag_data = capture_and_crop_from_bag_manual(bag_path)
    save_crop_cache(cache_dir, bag_data)
    print(f"Saved crop cache to: {cache_dir}")

    return bag_data


# =========================================================
# TOOTH DETECTION
# =========================================================
def get_detection_points_from_rgb_image(image_rgb: np.ndarray):
    """
    Roboflow tooth detector.
    Input must be RGB.
    """
    if not API_KEY:
        raise ValueError("ROBOFLOW_API_KEY is not set.")

    image_rgb = image_rgb.copy()

    # Correct conversion before JPEG encoding.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(".jpg", image_bgr)

    if not ok:
        raise ValueError("Failed to encode image for Roboflow request.")

    resp = requests.post(
        "https://serverless.roboflow.com/toothcariesdetection/2",
        params={"api_key": API_KEY},
        files={"file": ("image.jpg", encoded.tobytes(), "image/jpeg")},
        timeout=60,
    )

    resp.raise_for_status()
    result = resp.json()

    points = []

    for p in result.get("predictions", []):
        points.append({
            "cx": int(round(p["x"])),
            "cy": int(round(p["y"])),
            "class": p.get("class", "Tooth"),
            "confidence": float(p["confidence"]),
            "width": float(p.get("width", 0.0)),
            "height": float(p.get("height", 0.0)),
        })

    return points


# =========================================================
# SEGFORMER
# =========================================================
def load_segformer(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    image_processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model, image_processor, device


def get_gingival_mask_from_rgb_image(
    image_rgb: np.ndarray,
    model,
    image_processor,
    device
):
    pil_image = Image.fromarray(image_rgb)
    w, h = pil_image.size

    inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    mask = (predicted_mask > 0).astype(np.uint8) * 255

    return mask


# =========================================================
# GINGIVA COMPONENTS
# =========================================================
def split_upper_lower_mask(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        comp_mask = np.zeros_like(mask, dtype=np.uint8)
        comp_mask[labels == label_id] = 255

        cy = centroids[label_id][1]
        components.append((cy, comp_mask, area))

    if len(components) < 2:
        raise ValueError("Could not find two gingiva components in the mask.")

    components.sort(key=lambda x: x[0])

    upper_mask = components[0][1]
    lower_mask = components[1][1]

    return upper_mask, lower_mask


def split_upper_lower_points(points):
    if not points:
        return [], []

    ys = sorted(pt["cy"] for pt in points)
    median_y = ys[len(ys) // 2]

    upper_points = []
    lower_points = []

    for pt in points:
        if pt["cy"] < median_y:
            upper_points.append(pt)
        else:
            lower_points.append(pt)

    return upper_points, lower_points


def get_component_column_hits(component_mask, cx, x_tol=5):
    best = None
    best_dx = float("inf")

    h, w = component_mask.shape

    x_start = max(0, cx - x_tol)
    x_end = min(w, cx + x_tol + 1)

    for x in range(x_start, x_end):
        ys = np.where(component_mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        top_y = int(ys.min())
        bottom_y = int(ys.max())
        dx = abs(x - cx)

        if dx < best_dx:
            best_dx = dx
            best = (x, top_y, bottom_y)

    return best


def get_inner_outer_pairs(points, upper_mask, lower_mask, x_tol=5):
    upper_points, lower_points = split_upper_lower_points(points)

    results = []

    for pt in upper_points:
        hit = get_component_column_hits(upper_mask, pt["cx"], x_tol=x_tol)

        if hit is None:
            continue

        x_used, top_y, bottom_y = hit

        results.append({
            "group": "upper",
            "tooth_point": (int(pt["cx"]), int(pt["cy"])),
            "inner_point": (int(x_used), int(bottom_y)),
            "outer_point": (int(x_used), int(top_y)),
            "class": pt.get("class", "Tooth"),
            "confidence": float(pt["confidence"]),
            "det_width": float(pt.get("width", 0.0)),
            "det_height": float(pt.get("height", 0.0)),
        })

    for pt in lower_points:
        hit = get_component_column_hits(lower_mask, pt["cx"], x_tol=x_tol)

        if hit is None:
            continue

        x_used, top_y, bottom_y = hit

        results.append({
            "group": "lower",
            "tooth_point": (int(pt["cx"]), int(pt["cy"])),
            "inner_point": (int(x_used), int(top_y)),
            "outer_point": (int(x_used), int(bottom_y)),
            "class": pt.get("class", "Tooth"),
            "confidence": float(pt["confidence"]),
            "det_width": float(pt.get("width", 0.0)),
            "det_height": float(pt.get("height", 0.0)),
        })

    return results


# =========================================================
# PIXEL -> 3D -> MM
# =========================================================
def pixel_to_3d(u, v, depth_image, intr):
    h, w = depth_image.shape

    if not (0 <= u < w and 0 <= v < h):
        return None

    depth_raw = float(depth_image[v, u])

    if depth_raw <= 0:
        return None

    Z = depth_raw * intr["depth_scale"]
    X = (u - intr["ppx"]) * Z / intr["fx"]
    Y = (v - intr["ppy"]) * Z / intr["fy"]

    return np.array([X, Y, Z], dtype=np.float64)


def robust_pixel_to_3d(u, v, depth_image, intr, radius=DEPTH_SEARCH_RADIUS):
    p = pixel_to_3d(u, v, depth_image, intr)

    if p is not None:
        return p

    h, w = depth_image.shape
    best = None
    best_d = float("inf")

    for yy in range(max(0, v - radius), min(h, v + radius + 1)):
        for xx in range(max(0, u - radius), min(w, u + radius + 1)):
            candidate = pixel_to_3d(xx, yy, depth_image, intr)

            if candidate is None:
                continue

            d = abs(xx - u) + abs(yy - v)

            if d < best_d:
                best_d = d
                best = candidate

    return best


def distance_mm(p1, p2, depth_image, intr):
    P1 = robust_pixel_to_3d(
        int(p1[0]),
        int(p1[1]),
        depth_image,
        intr,
        radius=DEPTH_SEARCH_RADIUS
    )

    P2 = robust_pixel_to_3d(
        int(p2[0]),
        int(p2[1]),
        depth_image,
        intr,
        radius=DEPTH_SEARCH_RADIUS
    )

    if P1 is None or P2 is None:
        return None

    dist_m = np.linalg.norm(P1 - P2)

    return float(dist_m * 1000.0)


def add_mm_measurements_to_pairs(pairs, depth_image, intr):
    enriched = []

    for idx, pair in enumerate(pairs):
        tooth_pt = pair["tooth_point"]
        inner_pt = pair["inner_point"]
        outer_pt = pair["outer_point"]

        tooth_to_inner_mm = distance_mm(tooth_pt, inner_pt, depth_image, intr)
        tooth_to_outer_mm = distance_mm(tooth_pt, outer_pt, depth_image, intr)
        inner_to_outer_mm = distance_mm(inner_pt, outer_pt, depth_image, intr)

        pair2 = dict(pair)
        pair2["index"] = int(idx)
        pair2["tooth_to_inner_mm"] = tooth_to_inner_mm
        pair2["tooth_to_outer_mm"] = tooth_to_outer_mm
        pair2["inner_to_outer_mm"] = inner_to_outer_mm

        enriched.append(pair2)

    return enriched


# =========================================================
# TOOTH MAPPING
# =========================================================
def get_allowed_teeth_for_view(view):
    if view == "front":
        return FRONT_ALLOWED
    if view == "right":
        return RIGHT_ALLOWED
    if view == "left":
        return LEFT_ALLOWED
    return None


def estimate_midline_x(pairs):
    """
    Legacy helper kept for compatibility. The active front-view mapping below
    uses image/crop center x/y instead of estimated midline.
    """
    if not pairs:
        return None

    xs = sorted([p["tooth_point"][0] for p in pairs])
    n = len(xs)

    if n % 2 == 1:
        return float(xs[n // 2])

    return float(0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def split_by_midline_nearest_first(pairs, midline_x):
    """
    Legacy helper kept for compatibility. The active front-view mapping below
    uses quadrant sorting from image/crop center.
    """
    left = []
    right = []

    for p in pairs:
        x = p["tooth_point"][0]

        if x < midline_x:
            left.append(p)
        else:
            right.append(p)

    left = sorted(left, key=lambda p: abs(p["tooth_point"][0] - midline_x))
    right = sorted(right, key=lambda p: abs(p["tooth_point"][0] - midline_x))

    return left, right


def map_front_view_pairs(pairs, image_shape):
    """
    Front-view full-arch quadrant mapping using the crop center.

    Coordinate system:
      x increases left -> right
      y increases top -> bottom

    Quadrant logic:
      upper-left  nearest center -> 11, then 12,13,...18 outward
      upper-right nearest center -> 21, then 22,23,...28 outward
      lower-left  nearest center -> 41, then 42,43,...48 outward
      lower-right nearest center -> 31, then 32,33,...38 outward

    This assumes the manual crop is centered around the mouth midline.
    """
    if image_shape is None:
        raise ValueError("image_shape is required for front-view quadrant mapping.")

    h, w = image_shape[:2]
    center_x = w / 2.0
    center_y = h / 2.0

    upper_left = []
    upper_right = []
    lower_left = []
    lower_right = []

    for p in pairs:
        x, y = p["tooth_point"]

        if y < center_y:
            if x < center_x:
                upper_left.append(p)
            else:
                upper_right.append(p)
        else:
            if x < center_x:
                lower_left.append(p)
            else:
                lower_right.append(p)

    # Sort nearest to the center line first, then outward to the extreme teeth.
    # x-distance is the main ordering signal because FDI numbering progresses
    # horizontally away from the midline in a frontal image.
    upper_left = sorted(upper_left, key=lambda p: (abs(p["tooth_point"][0] - center_x), abs(p["tooth_point"][1] - center_y)))
    upper_right = sorted(upper_right, key=lambda p: (abs(p["tooth_point"][0] - center_x), abs(p["tooth_point"][1] - center_y)))
    lower_left = sorted(lower_left, key=lambda p: (abs(p["tooth_point"][0] - center_x), abs(p["tooth_point"][1] - center_y)))
    lower_right = sorted(lower_right, key=lambda p: (abs(p["tooth_point"][0] - center_x), abs(p["tooth_point"][1] - center_y)))

    upper_left_seq = ["11", "12", "13", "14", "15", "16", "17", "18"]
    upper_right_seq = ["21", "22", "23", "24", "25", "26", "27", "28"]
    lower_left_seq = ["41", "42", "43", "44", "45", "46", "47", "48"]
    lower_right_seq = ["31", "32", "33", "34", "35", "36", "37", "38"]

    mapping = {}

    for i, p in enumerate(upper_left):
        if i < len(upper_left_seq):
            mapping[p["index"]] = upper_left_seq[i]

    for i, p in enumerate(upper_right):
        if i < len(upper_right_seq):
            mapping[p["index"]] = upper_right_seq[i]

    for i, p in enumerate(lower_left):
        if i < len(lower_left_seq):
            mapping[p["index"]] = lower_left_seq[i]

    for i, p in enumerate(lower_right):
        if i < len(lower_right_seq):
            mapping[p["index"]] = lower_right_seq[i]

    return mapping


def center_align_sequence(sorted_pairs, sequence):
    n = len(sorted_pairs)
    m = len(sequence)

    if n <= 0:
        return {}

    if n >= m:
        used_pairs = sorted_pairs[:m]
        chosen = sequence
    else:
        offset = max(0, (m - n) // 2)
        used_pairs = sorted_pairs
        chosen = sequence[offset:offset + n]

    return {
        p["index"]: tooth
        for p, tooth in zip(used_pairs, chosen)
    }


def map_side_view_pairs(pairs, view):
    """
    Right / left side mapping.
    This is still a geometry prior, not perfect clinical tooth classification.
    """
    mapping = {}

    for group in ["upper", "lower"]:
        group_pairs = [p for p in pairs if p["group"] == group]
        group_pairs = sorted(group_pairs, key=lambda p: p["tooth_point"][0])

        if view == "right":
            if group == "upper":
                seq = ["18", "17", "16", "15", "14", "13", "12", "11"]
            else:
                seq = ["48", "47", "46", "45", "44", "43", "42", "41"]

        elif view == "left":
            if group == "upper":
                seq = ["21", "22", "23", "24", "25", "26", "27", "28"]
            else:
                seq = ["31", "32", "33", "34", "35", "36", "37", "38"]

        else:
            continue

        mapping.update(center_align_sequence(group_pairs, seq))

    return mapping


def filter_mapping_by_view(pairs, mapping, view):
    allowed = get_allowed_teeth_for_view(view)

    if allowed is None:
        return mapping

    filtered = {}

    for p in pairs:
        idx = p["index"]
        group = p["group"]
        tooth = mapping.get(idx)

        if tooth is None:
            continue

        if tooth in allowed.get(group, set()):
            filtered[idx] = tooth

    return filtered


def infer_tooth_mapping_for_view(pairs, view, image_shape=None):
    if view == "front":
        raw_map = map_front_view_pairs(pairs, image_shape=image_shape)
    elif view in ["left", "right"]:
        raw_map = map_side_view_pairs(pairs, view)
    else:
        # For unknown view, fall back to front-style quadrant mapping when possible.
        if image_shape is not None:
            raw_map = map_front_view_pairs(pairs, image_shape=image_shape)
        else:
            raise ValueError("image_shape is required when using front/unknown quadrant mapping.")

    filtered_map = filter_mapping_by_view(
        pairs,
        raw_map,
        view
    )

    return raw_map, filtered_map


def attach_tooth_ids_to_pairs(pairs, raw_map, filtered_map):
    out = []

    for p in pairs:
        q = dict(p)
        q["inferred_tooth_id_raw"] = raw_map.get(p["index"])
        q["inferred_tooth_id"] = filtered_map.get(p["index"])
        out.append(q)

    return out


# =========================================================
# JSON OUTPUT
# =========================================================
def make_json_safe(measured_pairs):
    out = []

    for pair in measured_pairs:
        out.append({
            "index": int(pair["index"]),
            "group": pair["group"],
            "class": pair.get("class", "Tooth"),
            "confidence": float(pair["confidence"]),
            "inferred_tooth_id": pair.get("inferred_tooth_id"),
            "inferred_tooth_id_raw": pair.get("inferred_tooth_id_raw"),
            "tooth_point": [
                int(pair["tooth_point"][0]),
                int(pair["tooth_point"][1])
            ],
            "inner_point": [
                int(pair["inner_point"][0]),
                int(pair["inner_point"][1])
            ],
            "outer_point": [
                int(pair["outer_point"][0]),
                int(pair["outer_point"][1])
            ],
            "tooth_to_inner_mm": (
                None if pair["tooth_to_inner_mm"] is None
                else float(pair["tooth_to_inner_mm"])
            ),
            "tooth_to_outer_mm": (
                None if pair["tooth_to_outer_mm"] is None
                else float(pair["tooth_to_outer_mm"])
            ),
            "inner_to_outer_mm": (
                None if pair["inner_to_outer_mm"] is None
                else float(pair["inner_to_outer_mm"])
            ),
        })

    return out


# =========================================================
# DRAWING
# =========================================================
def draw_component_masks(upper_mask, lower_mask):
    h, w = upper_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # RGB colors
    canvas[upper_mask > 0] = (0, 255, 0)
    canvas[lower_mask > 0] = (255, 0, 0)

    return canvas


def draw_inner_outer_pairs(mask, pairs):
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    for pair in pairs:
        idx = pair["index"]
        tooth_id = pair.get("inferred_tooth_id")
        kgw = safe_float(pair.get("inner_to_outer_mm"))

        tx, ty = pair["tooth_point"]
        ix, iy = pair["inner_point"]
        ox, oy = pair["outer_point"]

        cv2.circle(canvas, (tx, ty), 4, (255, 0, 0), -1)
        cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (ox, oy), 4, (255, 255, 0), -1)

        cv2.line(canvas, (ix, iy), (ox, oy), (0, 255, 0), 1)

        label = f"{idx}"

        if tooth_id:
            label = f"{tooth_id}"

        if kgw is not None:
            label += f":{kgw:.1f}"

        cv2.putText(
            canvas,
            label,
            (tx + 5, ty - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return canvas


def draw_inner_outer_pairs_on_rgb(image_rgb, pairs):
    canvas = image_rgb.copy()

    for pair in pairs:
        idx = pair["index"]
        tooth_id = pair.get("inferred_tooth_id")
        kgw = safe_float(pair.get("inner_to_outer_mm"))

        tx, ty = pair["tooth_point"]
        ix, iy = pair["inner_point"]
        ox, oy = pair["outer_point"]

        cv2.circle(canvas, (tx, ty), 4, (255, 0, 0), -1)
        cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (ox, oy), 4, (255, 255, 0), -1)

        cv2.line(canvas, (ix, iy), (ox, oy), (0, 255, 0), 2)

        cv2.putText(
            canvas,
            f"T{idx}",
            (tx + 4, ty - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"I{idx}",
            (ix + 4, iy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"O{idx}",
            (ox + 4, oy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        label = str(idx)

        if tooth_id:
            label = str(tooth_id)

        if kgw is not None:
            label += f":{kgw:.1f}mm"

        cv2.putText(
            canvas,
            label,
            (tx + 4, ty + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return canvas


# =========================================================
# MODE SELECTION
# =========================================================
def get_mode_sources(mode: str):
    if mode == "no_clahe":
        return {
            "detection_source": "rgb",
            "segmentation_source": "rgb",
        }

    if mode == "clahe_detection":
        return {
            "detection_source": "clahe",
            "segmentation_source": "rgb",
        }

    if mode == "clahe_segmentation":
        return {
            "detection_source": "rgb",
            "segmentation_source": "clahe",
        }

    if mode == "clahe_both":
        return {
            "detection_source": "clahe",
            "segmentation_source": "clahe",
        }

    raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(
    bag_path: str,
    output_dir: str,
    segformer_model_path: str,
    mode: str,
    crop_cache_dir: str = None,
    crop_mode: str = "manual",
    view: str = None,
):
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    if crop_mode != "manual":
        raise ValueError("This script is manual-only. Use --crop_mode manual.")

    ensure_dir(output_dir)

    if view is None:
        view = infer_view_from_bag_path(bag_path)

    base_name = Path(bag_path).stem

    print(f"Bag path   : {bag_path}")
    print(f"Output dir : {output_dir}")
    print(f"Mode       : {mode}")
    print(f"View       : {view}")

    # -----------------------------------------------------
    # Step 1: Get crop manually or from cache
    # -----------------------------------------------------
    print("\nStep 1: Capture/crop from bag using manual mode...")

    if crop_cache_dir:
        bag_data = get_or_create_manual_crop_cache(
            bag_path=bag_path,
            cache_dir=crop_cache_dir
        )
    else:
        bag_data = capture_and_crop_from_bag_manual(bag_path)

    rgb_crop = bag_data["rgb_crop"]
    depth_crop = bag_data["depth_crop"]
    depth_crop_vis = bag_data["depth_crop_vis"]
    intr = bag_data["intrinsics"]

    print("Loaded from cache:", bag_data.get("loaded_from_cache", False))

    # -----------------------------------------------------
    # Step 2: Prepare RGB and CLAHE
    # -----------------------------------------------------
    print("\nStep 2: Prepare RGB and CLAHE images...")

    rgb_crop_clahe = apply_clahe_rgb(
        rgb_crop,
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE
    )

    # Save base crops
    rgb_crop_path = os.path.join(output_dir, f"{base_name}_rgb_crop.png")
    rgb_crop_clahe_path = os.path.join(output_dir, f"{base_name}_rgb_crop_clahe.png")
    depth_crop_npy_path = os.path.join(output_dir, f"{base_name}_depth_crop.npy")
    depth_crop_vis_path = os.path.join(output_dir, f"{base_name}_depth_crop_vis.png")
    intrinsics_path = os.path.join(output_dir, f"{base_name}_intrinsics.json")

    save_rgb(rgb_crop_path, rgb_crop)
    save_rgb(rgb_crop_clahe_path, rgb_crop_clahe)
    np.save(depth_crop_npy_path, depth_crop)
    cv2.imwrite(depth_crop_vis_path, depth_crop_vis)
    save_json(intrinsics_path, intr)

    # -----------------------------------------------------
    # Step 3: Detection on both RGB and CLAHE
    # -----------------------------------------------------
    print("\nStep 3: Running tooth detection on RGB...")
    rgb_detection_points = get_detection_points_from_rgb_image(rgb_crop)
    print(f"RGB detections: {len(rgb_detection_points)}")

    print("Step 3b: Running tooth detection on CLAHE...")
    clahe_detection_points = get_detection_points_from_rgb_image(rgb_crop_clahe)
    print(f"CLAHE detections: {len(clahe_detection_points)}")

    rgb_detection_points_path = os.path.join(output_dir, f"{base_name}_rgb_detection_points.json")
    clahe_detection_points_path = os.path.join(output_dir, f"{base_name}_clahe_detection_points.json")

    save_json(rgb_detection_points_path, rgb_detection_points)
    save_json(clahe_detection_points_path, clahe_detection_points)

    # -----------------------------------------------------
    # Step 4: Load segmentation model
    # -----------------------------------------------------
    print("\nStep 4: Loading SegFormer...")
    seg_model, image_processor, device = load_segformer(segformer_model_path)

    # -----------------------------------------------------
    # Step 5: Segmentation on both RGB and CLAHE
    # -----------------------------------------------------
    print("\nStep 5: Running gingiva segmentation on RGB...")
    rgb_mask = get_gingival_mask_from_rgb_image(
        rgb_crop,
        seg_model,
        image_processor,
        device
    )

    print("Step 5b: Running gingiva segmentation on CLAHE...")
    clahe_mask = get_gingival_mask_from_rgb_image(
        rgb_crop_clahe,
        seg_model,
        image_processor,
        device
    )

    rgb_mask_path = os.path.join(output_dir, f"{base_name}_rgb_mask.png")
    clahe_mask_path = os.path.join(output_dir, f"{base_name}_clahe_mask.png")

    cv2.imwrite(rgb_mask_path, rgb_mask)
    cv2.imwrite(clahe_mask_path, clahe_mask)

    # -----------------------------------------------------
    # Step 6: Select active mode
    # -----------------------------------------------------
    print("\nStep 6: Selecting active mode sources...")

    mode_sources = get_mode_sources(mode)

    if mode_sources["detection_source"] == "rgb":
        active_points = rgb_detection_points
    else:
        active_points = clahe_detection_points

    if mode_sources["segmentation_source"] == "rgb":
        active_mask = rgb_mask
    else:
        active_mask = clahe_mask

    mode_sources_path = os.path.join(output_dir, f"{base_name}_mode_sources.json")
    save_json(
        mode_sources_path,
        {
            "mode": mode,
            "view": view,
            **mode_sources,
            "rgb_detection_count": len(rgb_detection_points),
            "clahe_detection_count": len(clahe_detection_points),
            "active_detection_count": len(active_points),
            "loaded_from_crop_cache": bag_data.get("loaded_from_cache", False),
            "crop_cache_dir": crop_cache_dir,
        }
    )

    print("Detection source:", mode_sources["detection_source"])
    print("Segmentation source:", mode_sources["segmentation_source"])
    print("Active detection count:", len(active_points))

    # -----------------------------------------------------
    # Step 7: Split upper/lower active mask
    # -----------------------------------------------------
    print("\nStep 7: Splitting upper/lower gingiva masks...")

    upper_mask, lower_mask = split_upper_lower_mask(active_mask)

    # -----------------------------------------------------
    # Step 8: Find inner/outer pairs
    # -----------------------------------------------------
    print("\nStep 8: Finding inner/outer border points...")

    pairs = get_inner_outer_pairs(
        active_points,
        upper_mask,
        lower_mask,
        x_tol=X_TOL
    )

    print(f"Pairs found: {len(pairs)}")

    # -----------------------------------------------------
    # Step 9: Convert to mm
    # -----------------------------------------------------
    print("\nStep 9: Converting distances to mm...")

    measured_pairs = add_mm_measurements_to_pairs(
        pairs,
        depth_crop,
        intr
    )

    # -----------------------------------------------------
    # Step 10: Tooth mapping
    # -----------------------------------------------------
    print("\nStep 10: Inferring tooth IDs...")

    raw_map, filtered_map = infer_tooth_mapping_for_view(
        measured_pairs,
        view=view,
        image_shape=rgb_crop.shape
    )

    measured_pairs = attach_tooth_ids_to_pairs(
        measured_pairs,
        raw_map,
        filtered_map
    )

    print("Raw mapping:", raw_map)
    print("Filtered mapping:", filtered_map)

    for pair in measured_pairs:
        print(f"\n[{pair['index']}] {pair['group']} tooth={pair.get('inferred_tooth_id')}")
        print("  tooth point:", pair["tooth_point"])
        print("  inner point:", pair["inner_point"])
        print("  outer point:", pair["outer_point"])
        print("  inner -> outer KGW (mm):", pair["inner_to_outer_mm"])

    # -----------------------------------------------------
    # Step 11: Draw results
    # -----------------------------------------------------
    print("\nStep 11: Drawing final outputs...")

    components_vis = draw_component_masks(upper_mask, lower_mask)
    final_mask_overlay = draw_inner_outer_pairs(active_mask, measured_pairs)

    # Main visualization: original RGB, always.
    final_rgb_overlay = draw_inner_outer_pairs_on_rgb(
        rgb_crop,
        measured_pairs
    )

    # Debug visualization: CLAHE RGB.
    final_rgb_clahe_overlay = draw_inner_outer_pairs_on_rgb(
        rgb_crop_clahe,
        measured_pairs
    )

    # -----------------------------------------------------
    # Step 12: Save final outputs
    # -----------------------------------------------------
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    upper_path = os.path.join(output_dir, f"{base_name}_upper_mask.png")
    lower_path = os.path.join(output_dir, f"{base_name}_lower_mask.png")
    components_path = os.path.join(output_dir, f"{base_name}_components_vis.png")
    mask_overlay_path = os.path.join(output_dir, f"{base_name}_inner_outer_pairs_indexed_mask.png")
    rgb_overlay_path = os.path.join(output_dir, f"{base_name}_inner_outer_pairs_indexed_rgb.png")
    rgb_clahe_overlay_path = os.path.join(output_dir, f"{base_name}_inner_outer_pairs_indexed_rgb_clahe.png")
    pairs_json_path = os.path.join(output_dir, f"{base_name}_pairs_measurements.json")
    tooth_mapping_path = os.path.join(output_dir, f"{base_name}_tooth_mapping.json")
    run_metadata_path = os.path.join(output_dir, f"{base_name}_run_metadata.json")

    cv2.imwrite(mask_path, active_mask)
    cv2.imwrite(upper_path, upper_mask)
    cv2.imwrite(lower_path, lower_mask)

    save_rgb(components_path, components_vis)
    save_rgb(mask_overlay_path, final_mask_overlay)
    save_rgb(rgb_overlay_path, final_rgb_overlay)
    save_rgb(rgb_clahe_overlay_path, final_rgb_clahe_overlay)

    save_json(pairs_json_path, make_json_safe(measured_pairs))
    save_json(
        tooth_mapping_path,
        {
            "view": view,
            "raw_mapping": raw_map,
            "filtered_mapping": filtered_map,
        }
    )

    save_json(
        run_metadata_path,
        {
            "bag_path": bag_path,
            "base_name": base_name,
            "output_dir": output_dir,
            "mode": mode,
            "view": view,
            "crop_mode": crop_mode,
            "crop_cache_dir": crop_cache_dir,
            "loaded_from_crop_cache": bag_data.get("loaded_from_cache", False),
            "roi": bag_data.get("roi"),
            "mode_sources": mode_sources,
            "rgb_detection_count": len(rgb_detection_points),
            "clahe_detection_count": len(clahe_detection_points),
            "active_detection_count": len(active_points),
            "pairs_found": len(measured_pairs),
        }
    )

    print("\nSaved outputs:")
    print("RGB crop:", rgb_crop_path)
    print("CLAHE crop:", rgb_crop_clahe_path)
    print("Depth crop:", depth_crop_npy_path)
    print("Depth visualization:", depth_crop_vis_path)
    print("Intrinsics:", intrinsics_path)
    print("RGB detection points:", rgb_detection_points_path)
    print("CLAHE detection points:", clahe_detection_points_path)
    print("RGB mask:", rgb_mask_path)
    print("CLAHE mask:", clahe_mask_path)
    print("Active mask:", mask_path)
    print("Upper mask:", upper_path)
    print("Lower mask:", lower_path)
    print("Components:", components_path)
    print("Mask overlay:", mask_overlay_path)
    print("RGB overlay:", rgb_overlay_path)
    print("CLAHE RGB overlay:", rgb_clahe_overlay_path)
    print("Pairs JSON:", pairs_json_path)
    print("Tooth mapping:", tooth_mapping_path)
    print("Mode sources:", mode_sources_path)
    print("Run metadata:", run_metadata_path)

    return {
        "pairs": measured_pairs,
        "raw_mapping": raw_map,
        "filtered_mapping": filtered_map,
        "paths": {
            "pairs_json": pairs_json_path,
            "rgb_overlay": rgb_overlay_path,
            "run_metadata": run_metadata_path,
        }
    }


# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bag_path",
        default=DEFAULT_BAG_PATH,
        help="Path to RealSense .bag file."
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files."
    )

    parser.add_argument(
        "--segformer_model_path",
        default=DEFAULT_SEGFORMER_MODEL_PATH,
        help="Path to local SegFormer model folder."
    )

    parser.add_argument(
        "--mode",
        default="clahe_both",
        choices=VALID_MODES,
        help="Experiment mode."
    )

    parser.add_argument(
        "--crop_mode",
        default="manual",
        choices=["manual"],
        help="Manual only. Kept for compatibility with batch script."
    )

    parser.add_argument(
        "--crop_cache_dir",
        default=None,
        help="Folder to save/load cached manual crop. If provided, crop is only done once per bag."
    )

    parser.add_argument(
        "--view",
        default=None,
        choices=["front", "right", "left", "unknown"],
        help="Optional view override. If omitted, view is inferred from bag filename."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        segformer_model_path=args.segformer_model_path,
        mode=args.mode,
        crop_cache_dir=args.crop_cache_dir,
        crop_mode=args.crop_mode,
        view=args.view,
    )


if __name__ == "__main__":
    main()