import base64
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import aiplatform

from backend.config.settings import (
    GCP_ENDPOINT_ID,
    GCP_ENDPOINT_IMAGE_MAX_SIZE,
    GCP_PROJECT_ID,
    GCP_REGION,
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


def confidence_logic(mask):
    return round(float(np.mean(mask)), 2)
