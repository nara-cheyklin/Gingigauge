import os
from dotenv import load_dotenv

load_dotenv()

# settings.py

# Application project: hosts Cloud Run, Firestore, Secret Manager.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")

# Vertex AI project: where the deployed model/endpoint lives. Falls back to
# GCP_PROJECT_ID if not set, so single-project local dev still works.
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", GCP_PROJECT_ID)
VERTEX_REGION = os.getenv("VERTEX_REGION", GCP_REGION)

GCP_ENDPOINT_ID = os.getenv("GCP_ENDPOINT_ID")
GCP_ENDPOINT_IMAGE_MAX_SIZE = int(os.getenv("GCP_ENDPOINT_IMAGE_MAX_SIZE", "0"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ALLOWED_FILE_TYPES = [
    "application/octet-stream",
    "application/x-bag"
]

KGW_THRESHOLD_MM = 2.0

# RealSense D405 intrinsics
# Replace these with your actual calibrated values
# CAMERA_INTRINSICS = {
#     "fx": 430.0,
#     "fy": 430.0,
#     "cx": 320.0,
#     "cy": 240.0
# }

# Depth scale:
# If depth image is uint16 in millimeters, use 1.0
# If depth is in meters, use 1000.0 when converting to mm
DEPTH_UNIT_SCALE = 1.0