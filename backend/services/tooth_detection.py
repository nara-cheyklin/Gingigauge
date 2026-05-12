"""Roboflow tooth detection."""

import cv2
import numpy as np
import requests

from backend.config.settings import ROBOFLOW_API_KEY

_ROBOFLOW_URL = "https://serverless.roboflow.com/toothcariesdetection/2"


def detect_teeth(image_rgb: np.ndarray) -> list:
    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY is not set.")

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to encode image for Roboflow.")

    resp = requests.post(
        _ROBOFLOW_URL,
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
