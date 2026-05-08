import base64
import os
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.full_pipeline import run_full_pipeline
from backend.services.rosbag_processing import (
    extract_rgb_and_depth_from_rosbag,
    extract_camera_intrinsics_from_rosbag,
    cv2_to_bytes,
)
from backend.services.rosbag_upload import (
    generate_upload_url,
    download_rosbag_to_temp,
    delete_rosbag,
)

router = APIRouter()


# --- /predict/rosbag/upload-url ---------------------------------------------
# Step 1 of the rosbag flow. The browser asks for a v4 signed URL it can PUT
# the .bag straight to GCS with. This sidesteps Cloud Run's 32 MiB
# request-body limit.

class RosbagUploadUrlRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=200)
    content_type: str | None = Field(default="application/octet-stream", max_length=120)


@router.post("/predict/rosbag/upload-url")
async def predict_rosbag_upload_url(payload: RosbagUploadUrlRequest):
    if not payload.filename.lower().endswith(".bag"):
        raise HTTPException(status_code=400, detail="Only .bag files are accepted.")

    try:
        return generate_upload_url(
            filename=payload.filename,
            content_type=payload.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {e}")


# --- /predict/rosbag/preview ------------------------------------------------
# Step 1.5 of the rosbag flow (optional ROI crop UI).
# The frontend calls this right after the GCS upload to get a JPEG preview of
# the first RGB frame.  The GCS object is intentionally NOT deleted here --
# it is still needed for /predict/rosbag.

class RosbagPreviewRequest(BaseModel):
    gcs_path: str = Field(..., min_length=1, max_length=500)


@router.post("/predict/rosbag/preview")
async def predict_rosbag_preview(payload: RosbagPreviewRequest):
    try:
        rosbag_path = download_rosbag_to_temp(payload.gcs_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch upload: {e}")

    try:
        rgb_frame, _ = extract_rgb_and_depth_from_rosbag(rosbag_path)
        image_bytes = cv2_to_bytes(rgb_frame)
        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(rosbag_path):
            try:
                os.remove(rosbag_path)
            except OSError:
                pass
        # Do NOT delete the GCS object -- the bag is still needed for prediction.


# --- /predict/rosbag --------------------------------------------------------
# Step 2 of the rosbag flow. The frontend has already PUT the .bag to GCS;
# now it sends only the object path here. The backend pulls the bag from GCS,
# runs the full pipeline, and streams the annotated JPEG back with the KGW
# metrics in response headers.
# Optional roi=[x,y,w,h] crops both the RGB frame and depth map before the
# pipeline runs, and shifts the camera principal point accordingly.

class RosbagPredictRequest(BaseModel):
    gcs_path: str = Field(..., min_length=1, max_length=500)
    roi: list[int] | None = Field(
        default=None,
        description="Optional crop region [x, y, w, h] in image pixels.",
    )


@router.post("/predict/rosbag")
async def predict_rosbag(payload: RosbagPredictRequest):
    try:
        rosbag_path = download_rosbag_to_temp(payload.gcs_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch upload: {e}")

    try:
        rgb_frame, depth_frame = extract_rgb_and_depth_from_rosbag(rosbag_path)
        depth_intrinsics = extract_camera_intrinsics_from_rosbag(rosbag_path)

        # Apply ROI crop if the frontend sent one
        if payload.roi and len(payload.roi) == 4:
            x, y, w, h = payload.roi
            rgb_frame   = rgb_frame[y:y + h, x:x + w]
            depth_frame = depth_frame[y:y + h, x:x + w]
            depth_intrinsics = dict(depth_intrinsics)
            depth_intrinsics["ppx"] = depth_intrinsics["ppx"] - x
            depth_intrinsics["ppy"] = depth_intrinsics["ppy"] - y

        image_bytes = cv2_to_bytes(rgb_frame)
        result = run_full_pipeline(
            image_bytes=image_bytes,
            depth_map=depth_frame,
            depth_intrinsics=depth_intrinsics,
        )

        jpg_bytes = base64.b64decode(result["image_base64"])

        headers = {
            "X-KGW-MM": str(result["kgw_mm"]) if result["kgw_mm"] is not None else "null",
            "X-Confidence": str(result["confidence"]) if result["confidence"] is not None else "null",
            "X-Interpretation": result["interpretation"],
            "Access-Control-Expose-Headers": "X-KGW-MM, X-Confidence, X-Interpretation",
        }

        return StreamingResponse(
            BytesIO(jpg_bytes),
            media_type="image/jpeg",
            headers=headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Local temp file cleanup
        if os.path.exists(rosbag_path):
            try:
                os.remove(rosbag_path)
            except OSError:
                pass
        # Best-effort GCS cleanup so we don't accumulate large rosbag uploads
        delete_rosbag(payload.gcs_path)
