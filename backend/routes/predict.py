from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

from backend.config.settings import ALLOWED_FILE_TYPES, ALLOWED_IMAGE_FILE_TYPES
from backend.services.rosbag_processing import extract_rgb_and_depth_from_rosbag, cv2_to_bytes
from backend.services.inference import run_inference_rgb_only
from backend.services.full_pipeline import run_full_pipeline

router = APIRouter()

def save_temp_file(file: UploadFile):
    suffix = os.path.splitext(file.filename)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return tmp.name


@router.post("/predict")
@router.post("/predict/rgb")
async def predict_rgb_only(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image file type")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        result = run_inference_rgb_only(image_bytes=image_bytes)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/rosbag")
async def predict_rosbag(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    rosbag_path = save_temp_file(file)

    try:
        rgb_frame, depth_frame = extract_rgb_and_depth_from_rosbag(rosbag_path)

        image_bytes = cv2_to_bytes(rgb_frame)

        result = run_full_pipeline(
            image_bytes=image_bytes,
            depth_map=depth_frame
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(rosbag_path):
            os.remove(rosbag_path)
