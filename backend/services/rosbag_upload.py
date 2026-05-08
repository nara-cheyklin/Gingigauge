"""
Direct-to-GCS rosbag upload helpers.

Cloud Run rejects HTTP request bodies over 32 MiB. Rosbags are typically
100+ MiB, so we cannot accept them as multipart uploads. Instead the frontend
asks the backend for a v4 signed URL, PUTs the .bag straight to GCS, and then
calls /predict/rosbag with the resulting object path.

Signed-URL generation on Cloud Run requires IAM-based signing because the
metadata server's credentials do not expose a private key. The runtime
service account therefore needs `roles/iam.serviceAccountTokenCreator` on
itself; the deployment guide grants this.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import timedelta
from uuid import uuid4

from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage

from backend.config.settings import GCP_PROJECT_ID, GCS_BUCKET_NAME

UPLOAD_PREFIX = "uploads"
SIGNED_URL_EXPIRY = timedelta(minutes=15)
DEFAULT_CONTENT_TYPE = "application/octet-stream"

_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=GCP_PROJECT_ID)
    return _storage_client


def _signing_credentials():
    """
    Refresh the runtime service-account credentials and return them. Used to
    sign URLs via IAM (the metadata server's token alone can't sign blobs).
    """
    creds, _ = google_auth_default()
    creds.refresh(GoogleAuthRequest())
    return creds


def _sanitize_filename(name: str) -> str:
    """
    Strip directories and dangerous characters from a user-supplied filename
    before it becomes part of an object path.
    """
    base = os.path.basename(name or "").strip() or "upload.bag"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def generate_upload_url(
    filename: str,
    content_type: str = DEFAULT_CONTENT_TYPE,
) -> dict:
    """
    Issue a v4 signed URL the browser can PUT a rosbag to. Returns the URL
    plus the GCS object path the frontend should later send to /predict/rosbag.
    """
    safe = _sanitize_filename(filename)
    blob_path = f"{UPLOAD_PREFIX}/{uuid4().hex}/{safe}"

    blob = _get_storage_client().bucket(GCS_BUCKET_NAME).blob(blob_path)
    creds = _signing_credentials()

    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=SIGNED_URL_EXPIRY,
        method="PUT",
        content_type=content_type,
        service_account_email=creds.service_account_email,
        access_token=creds.token,
    )

    return {
        "upload_url": upload_url,
        "gcs_path": blob_path,
        "expires_in_seconds": int(SIGNED_URL_EXPIRY.total_seconds()),
        "content_type": content_type,
    }


def _validate_gcs_path(gcs_path: str) -> str:
    """
    Make sure the path the frontend supplies points inside our upload prefix
    and not somewhere else in the bucket.
    """
    if not gcs_path or not isinstance(gcs_path, str):
        raise ValueError("gcs_path is required.")
    cleaned = gcs_path.lstrip("/")
    if not cleaned.startswith(f"{UPLOAD_PREFIX}/"):
        raise ValueError("gcs_path must reference the uploads/ prefix.")
    if ".." in cleaned.split("/"):
        raise ValueError("gcs_path may not contain '..'.")
    return cleaned


def download_rosbag_to_temp(gcs_path: str) -> str:
    """
    Download the rosbag from GCS to a local temp file and return its path.
    Caller is responsible for deleting the file when done.
    """
    cleaned = _validate_gcs_path(gcs_path)
    blob = _get_storage_client().bucket(GCS_BUCKET_NAME).blob(cleaned)
    if not blob.exists():
        raise FileNotFoundError(f"Object not found in bucket: {cleaned}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bag")
    tmp.close()  # we just want the path
    blob.download_to_filename(tmp.name)
    return tmp.name


def delete_rosbag(gcs_path: str) -> None:
    """Best-effort cleanup of an uploaded rosbag after processing."""
    try:
        cleaned = _validate_gcs_path(gcs_path)
        blob = _get_storage_client().bucket(GCS_BUCKET_NAME).blob(cleaned)
        blob.delete(if_generation_match=None)
    except Exception:
        # Cleanup is non-critical; never fail a request because we couldn't tidy up.
        pass
