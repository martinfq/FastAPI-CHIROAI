import os
import re
from datetime import date, datetime
from functools import lru_cache

import cv2
import numpy as np
from paddleocr import PaddleOCR

from app.models.ocr import OCRResult

os.environ["FLAGS_log_level"] = "3"

# Crop region for this receipt layout (y1, y2, x1, x2)
_CROP = (130, 480, 40, 350)

_MONTHS: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3,
    "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}


@lru_cache(maxsize=1)
def _get_engine() -> PaddleOCR:
    return PaddleOCR(lang="es")


def warmup() -> None:
    """Load the OCR model into memory. Call once at startup to avoid cold start."""
    _get_engine()


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes and apply preprocessing pipeline entirely in memory."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    y1, y2, x1, x2 = _CROP
    crop = img[y1:y2, x1:x2]

    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # PaddleOCR expects BGR (H, W, 3)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _parse_date(text: str) -> date | None:
    match = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", text)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _MONTHS.get(month_name)
    if not month:
        return None
    return datetime(int(year), month, int(day)).date()


def extract_receipt_data(image_bytes: bytes) -> OCRResult:
    """Full pipeline: raw image bytes → structured receipt data."""
    img = _preprocess(image_bytes)
    prediction = _get_engine().predict(img)
    texts = [t.strip().lower() for t in prediction[0]["rec_texts"]]

    valor: float | None = None
    fecha: date | None = None
    banco_destino: str | None = None
    comprobante: str | None = None

    for t in texts:
        if "$" in t and valor is None:
            try:
                valor = float(t.replace("$", "").replace(",", "."))
            except ValueError:
                pass

    for t in texts:
        if fecha is None and "de" in t and any(c.isdigit() for c in t):
            fecha = _parse_date(t)

    for i, t in enumerate(texts):
        if "banco destino" in t and i + 1 < len(texts):
            banco_destino = texts[i + 1]
            break

    for i, t in enumerate(texts):
        if "comprobante" in t and i + 1 < len(texts):
            comprobante = texts[i + 1]
            break

    return OCRResult(valor=valor, fecha=fecha, banco_destino=banco_destino, comprobante=comprobante)
