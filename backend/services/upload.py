import base64
from datetime import datetime, timezone

from google.cloud import storage, firestore

from backend.config.settings import GCP_PROJECT_ID, GCS_BUCKET_NAME

_storage_client = None
_firestore_client = None


def _get_storage():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=GCP_PROJECT_ID)
    return _storage_client


def _get_firestore():
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client(project=GCP_PROJECT_ID)
    return _firestore_client


def upload_result(patient_id: str, image_data_url: str) -> dict:
    _, b64 = image_data_url.split(",", 1)
    jpg_bytes = base64.b64decode(b64)

    ts = datetime.now(timezone.utc)
    blob_name = f"results/{patient_id}/{ts.strftime('%Y%m%d_%H%M%S')}.jpg"

    bucket = _get_storage().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(jpg_bytes, content_type="image/jpeg")

    image_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_name}"

    db = _get_firestore()
    doc_ref = db.collection("patient_records").document()
    doc_ref.set({
        "patient_id": patient_id,
        "image_url": image_url,
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    return {
        "record_id": doc_ref.id,
        "image_url": image_url,
    }
