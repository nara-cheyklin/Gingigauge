from pathlib import Path
from PIL import Image


def validate_file_exists(image_path: str) -> None:
    if not Path(image_path).exists():
        raise ValueError("Image file does not exist.")


def validate_image_readable(image_path: str) -> None:
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        raise ValueError("Invalid or corrupted image file.") from e


def check_image_quality(image_path: str) -> dict:
    """
    Placeholder quality check.
    Later you can add blur detection, brightness check, framing check, etc.
    """
    validate_file_exists(image_path)
    validate_image_readable(image_path)

    return {
        "passed": True,
        "message": "Image quality acceptable."
    }