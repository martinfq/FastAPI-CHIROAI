import cv2
import numpy as np

# Crop region for this receipt layout (y1, y2, x1, x2)
_CROP = (130, 480, 40, 350)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes and apply preprocessing pipeline entirely in memory."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")

    y1, y2, x1, x2 = _CROP
    crop = img[y1:y2, x1:x2]

    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # RapidOCR accepts OpenCV-style BGR arrays.
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
