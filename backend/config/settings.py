import os
from dotenv import load_dotenv

load_dotenv()

# settings.py

MODEL_PATH = "model.pth"

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_ENDPOINT_ID = os.getenv("GCP_ENDPOINT_ID")
GCP_ENDPOINT_IMAGE_MAX_SIZE = int(os.getenv("GCP_ENDPOINT_IMAGE_MAX_SIZE", "0"))

ALLOWED_FILE_TYPES = [
    "application/octet-stream",
    "application/x-bag"
]

ALLOWED_IMAGE_FILE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

KGW_THRESHOLD_MM = 2.0

# RealSense D405 intrinsics
# Replace these with your actual calibrated values
CAMERA_INTRINSICS = {
    "fx": 430.0,
    "fy": 430.0,
    "cx": 320.0,
    "cy": 240.0
}

# Depth scale:
# If depth image is uint16 in millimeters, use 1.0
# If depth is in meters, use 1000.0 when converting to mm
DEPTH_UNIT_SCALE = 1.0
