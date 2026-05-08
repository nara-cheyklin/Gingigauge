import base64
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import aiplatform

from backend.config.settings import (
    GCP_ENDPOINT_ID,
    GCP_ENDPOINT_IMAGE_MAX_SIZE,
    VERTEX_PROJECT_ID,
    VERTEX_REGION,
)

_endpoint = None


def get_endpoint():
    global _endpoint

    if _endpoint is None:
        aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
        _endpoint = aiplatform.Endpoint(
            endpoint_name=(
                f"projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_REGION}/"
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


def _normalize_confidence(value):
    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence > 1.0:
        confidence /= 100.0

    return round(max(0.0, min(confidence, 1.0)), 4)


def prediction_to_result(prediction):
    if "mask_b64" not in prediction:
        raise RuntimeError("Vertex AI prediction did not include mask_b64")

    mask_bytes = base64.b64decode(prediction["mask_b64"])
    mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
    mask_np = np.array(mask_img)

    confidence = None
    for key in ("model_confidence", "confidence", "score", "probability"):
        confidence = _normalize_confidence(prediction.get(key))
        if confidence is not None:
            break

    return {
        "mask": (mask_np > 127).astype(np.uint8),
        "confidence": confidence,
    }


def predict_with_endpoint(image_bytes):
    endpoint = get_endpoint()
    image_b64 = image_bytes_to_endpoint_b64(image_bytes)
    response = endpoint.predict(instances=[{"b64": image_b64}])

    if not response.predictions:
        raise RuntimeError("Vertex AI endpoint returned no predictions")

    return prediction_to_result(response.predictions[0])


def resize_mask_to_original(mask, original_size):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)
