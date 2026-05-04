from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.upload import upload_result as _upload_result

router = APIRouter()


class UploadRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=100)
    image_data_url: str = Field(..., min_length=1)


@router.post("/upload")
async def upload_to_cloud(payload: UploadRequest):
    try:
        result = _upload_result(
            patient_id=payload.patient_id,
            image_data_url=payload.image_data_url,
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
