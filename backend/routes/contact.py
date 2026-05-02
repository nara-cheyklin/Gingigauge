from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from backend.services.contact import save_contact_message

router = APIRouter()


class ContactMessage(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=5000)


@router.post("/contact")
async def create_contact_message(payload: ContactMessage):
    try:
        message_id = save_contact_message(payload.model_dump())

        return {
            "success": True,
            "id": message_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
