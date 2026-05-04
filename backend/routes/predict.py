import base64
import tempfile
import os
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from backend.config.settings import ALLOWED_FILE_TYPES
from backend.services.rosbag_processing import extract_rgb_and_depth_from_rosbag, cv2_to_bytes
from backend.services.full_pipeline import run_full_pipeline

router = APIRouter()


def save_temp_file(file: UploadFile):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return tmp.name


@router.post("/predict/rosbag")
async def predict_rosbag(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only .bag files are accepted.")

    rosbag_path = save_temp_file(file)

    try:
        rgb_frame, depth_frame = extract_rgb_and_depth_from_rosbag(rosbag_path)
        image_bytes = cv2_to_bytes(rgb_frame)
        result = run_full_pipeline(image_bytes=image_bytes, depth_map=depth_frame)

        jpg_bytes = base64.b64decode(result["image_base64"])

        headers = {
            "X-KGW-MM": str(result["kgw_mm"]) if result["kgw_mm"] is not None else "null",
            "X-Confidence": str(result["confidence"]),
            "X-Interpretation": result["interpretation"],
            "Access-Control-Expose-Headers": "X-KGW-MM, X-Confidence, X-Interpretation",
        }

        return StreamingResponse(
            BytesIO(jpg_bytes),
            media_type="image/jpeg",
            headers=headers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(rosbag_path):
            os.remove(rosbag_path)
