import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from google.cloud import aiplatform

from backend.config.settings import (
    CAMERA_INTRINSICS,
    DEPTH_UNIT_SCALE,
    GCP_ENDPOINT_ID,
    GCP_ENDPOINT_IMAGE_MAX_SIZE,
    GCP_PROJECT_ID,
    GCP_REGION,
    KGW_THRESHOLD_MM,
)

_endpoint = None

def get_endpoint():
    global _endpoint

    if _endpoint is None:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        _endpoint = aiplatform.Endpoint(
            endpoint_name=(
                f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/"
                f"endpoints/{GCP_ENDPOINT_ID}"
            )
        )

    return _endpoint


def image_bytes_to_endpoint_b64(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    if GCP_ENDPOINT_IMAGE_MAX_SIZE:
        image.thumbnail((GCP_ENDPOINT_IMAGE_MAX_SIZE, GCP_ENDPOINT_IMAGE_MAX_SIZE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prediction_to_mask(prediction):
    if "mask_b64" not in prediction:
        raise RuntimeError("Vertex AI prediction did not include mask_b64")

    mask_bytes = base64.b64decode(prediction["mask_b64"])
    mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
    mask_np = np.array(mask_img)

    return (mask_np > 127).astype(np.uint8)


def predict_mask_with_endpoint(image_bytes):
    endpoint = get_endpoint()
    image_b64 = image_bytes_to_endpoint_b64(image_bytes)
    response = endpoint.predict(instances=[{"b64": image_b64}])

    if not response.predictions:
        raise RuntimeError("Vertex AI endpoint returned no predictions")

    return prediction_to_mask(response.predictions[0])

def resize_mask_to_original(mask, original_size):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


# -------------------------------
# Find two measurement points
# Placeholder logic:
# choose topmost and bottommost mask pixels in the center column
# Replace later with your actual MGJ/KG boundary logic
# -------------------------------
def find_measurement_points(binary_mask):
    h, w = binary_mask.shape
    center_x = w // 2

    ys = np.where(binary_mask[:, center_x] > 0)[0]

    if len(ys) < 2:
        # fallback: search globally
        coords = np.column_stack(np.where(binary_mask > 0))
        if len(coords) < 2:
            raise RuntimeError("Could not find enough mask points for measurement")

        top = coords[np.argmin(coords[:, 0])]
        bottom = coords[np.argmax(coords[:, 0])]
        p1 = (int(top[1]), int(top[0]))      # (u, v)
        p2 = (int(bottom[1]), int(bottom[0]))
        return p1, p2

    p1 = (center_x, int(ys.min()))
    p2 = (center_x, int(ys.max()))
    return p1, p2


# -------------------------------
# Convert pixel + depth to 3D
# -------------------------------
def pixel_to_3d(u, v, depth_value, intrinsics):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    X = (u - cx) * depth_value / fx
    Y = (v - cy) * depth_value / fy
    Z = depth_value

    return np.array([X, Y, Z], dtype=np.float32)


# -------------------------------
# Compute real-world distance in mm
# -------------------------------
def calculate_kgw_from_depth(binary_mask, depth_map, intrinsics, depth_scale=1.0):
    p1, p2 = find_measurement_points(binary_mask)

    u1, v1 = p1
    u2, v2 = p2

    if not (0 <= v1 < depth_map.shape[0] and 0 <= u1 < depth_map.shape[1]):
        raise RuntimeError("Point 1 is outside depth map bounds")

    if not (0 <= v2 < depth_map.shape[0] and 0 <= u2 < depth_map.shape[1]):
        raise RuntimeError("Point 2 is outside depth map bounds")

    z1 = float(depth_map[v1, u1]) * depth_scale
    z2 = float(depth_map[v2, u2]) * depth_scale

    if z1 <= 0 or z2 <= 0:
        raise RuntimeError("Invalid depth values at measurement points")

    p3d_1 = pixel_to_3d(u1, v1, z1, intrinsics)
    p3d_2 = pixel_to_3d(u2, v2, z2, intrinsics)

    distance_mm = np.linalg.norm(p3d_2 - p3d_1)

    return round(float(distance_mm), 2), p1, p2


# -------------------------------
# Confidence placeholder
# -------------------------------
def confidence_logic(mask):
    return round(float(np.mean(mask)), 2)


def mask_to_base64_png(binary_mask):
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def overlay_mask_on_image(image_bytes, binary_mask):
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_pixels = np.array(overlay)
    mask_pixels = np.array(mask_img) > 127
    overlay_pixels[mask_pixels] = [0, 200, 255, 110]
    overlay = Image.fromarray(overlay_pixels, mode="RGBA")

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    buffer = BytesIO()
    combined.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -------------------------------
# Draw measurement annotations on the original image and return as base64 JPEG
# -------------------------------
def annotate_image(image_bytes, point_1, point_2, kgw_mm):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    r = max(4, image.width // 80)
    dot_color = (0, 255, 0)
    line_color = (0, 200, 255)

    draw.line([point_1, point_2], fill=line_color, width=max(2, r // 2))

    for pt in (point_1, point_2):
        x, y = pt
        draw.ellipse([x - r, y - r, x + r, y + r], fill=dot_color, outline=(0, 0, 0), width=1)

    label = f"KGW: {kgw_mm} mm"
    margin = 8
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 30))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    tx = max(margin, min(point_1[0], image.width - text_w - margin))
    ty = max(margin, point_1[1] - text_h - margin * 2)
    draw.rectangle([tx - margin, ty - margin, tx + text_w + margin, ty + text_h + margin], fill=(0, 0, 0, 180))
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -------------------------------
# RGB-only inference for testing the deployed segmentation endpoint
# -------------------------------
def run_inference_rgb_only(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_size = image.size  # (width, height)
    binary_mask_small = predict_mask_with_endpoint(image_bytes)
    binary_mask = resize_mask_to_original(binary_mask_small, original_size)

    return {
        "confidence": confidence_logic(binary_mask),
        "image_size": {
            "width": original_size[0],
            "height": original_size[1]
        },
        "mask_size": {
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0]
        },
        "mask_base64": mask_to_base64_png(binary_mask),
        "image_base64": overlay_mask_on_image(image_bytes, binary_mask)
    }
