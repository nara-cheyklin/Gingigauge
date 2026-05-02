from google.cloud import firestore

from backend.config.settings import GCP_PROJECT_ID

_db = None


def get_db():
    global _db

    if _db is None:
        _db = firestore.Client(project=GCP_PROJECT_ID)

    return _db


def save_contact_message(message):
    db = get_db()
    doc_ref = db.collection("contact_messages").document()
    doc_ref.set({
        **message,
        "status": "new",
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    return doc_ref.id
