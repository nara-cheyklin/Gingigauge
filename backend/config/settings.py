from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "model.pth"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)