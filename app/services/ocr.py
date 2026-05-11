from dataclasses import dataclass
import logging
import re
import unicodedata
from datetime import date, datetime
from functools import lru_cache

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from app.models.ocr import OCRResult

logger = logging.getLogger(__name__)

# Crop region for this receipt layout (y1, y2, x1, x2)
_CROP = (130, 480, 40, 350)

_MONTHS: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3,
    "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

_AMOUNT_RE = re.compile(r"\$\s*([\d.,]+)")
_DATE_RE = re.compile(r"(\d{1,2}) de (\w+) de (\d{4})")
_MOTIVO_LABELS = (
    "motivo",
    "concepto",
    "descripcion",
    "descripcion de la operacion",
    "detalle",
    "referencia",
)
_BANK_LABELS = ("banco destino", "banco de destino")
_RECEIPT_LABELS = ("comprobante", "numero de comprobante", "nro comprobante")


@dataclass(frozen=True)
class OCRExtraction:
    result: OCRResult
    texts: list[str]


@lru_cache(maxsize=1)
def _get_engine() -> RapidOCR:
    logger.info("Initializing RapidOCR engine")
    engine = RapidOCR()
    logger.info("RapidOCR engine ready")
    return engine


def warmup() -> None:
    """Load the OCR model into memory. Call once at startup to avoid cold start."""
    _get_engine()


def _preprocess(image_bytes: bytes) -> np.ndarray:
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


def _parse_date(text: str) -> date | None:
    normalized = _normalize_text(text)
    match = _DATE_RE.search(normalized)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _MONTHS.get(month_name)
    if not month:
        return None
    return datetime(int(year), month, int(day)).date()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents).strip().lower()


def _clean_texts(prediction: list[list[object]] | None) -> list[str]:
    return [str(line[1]).strip() for line in prediction or [] if len(line) > 1 and str(line[1]).strip()]


def _parse_amount(value: str) -> float | None:
    normalized = value.replace(" ", "")
    if not normalized:
        return None

    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        head, tail = normalized.rsplit(",", 1)
        normalized = f"{head.replace(',', '')}.{tail}" if len(tail) <= 2 else normalized.replace(",", "")
    elif "." in normalized:
        head, tail = normalized.rsplit(".", 1)
        normalized = normalized if len(tail) <= 2 else normalized.replace(".", "")

    try:
        return float(normalized)
    except ValueError:
        return None


def _extract_amount(texts: list[str]) -> float | None:
    for text in texts:
        match = _AMOUNT_RE.search(text)
        if not match:
            continue
        amount = _parse_amount(match.group(1))
        if amount is not None:
            return amount
    return None


def _extract_date(texts: list[str]) -> date | None:
    for text in texts:
        if any(char.isdigit() for char in text):
            parsed = _parse_date(text)
            if parsed is not None:
                return parsed
    return None


def _next_meaningful_text(texts: list[str], start_index: int) -> str | None:
    for next_index in range(start_index, len(texts)):
        candidate = texts[next_index].strip()
        if candidate:
            return candidate
    return None


def _extract_labeled_value(texts: list[str], labels: tuple[str, ...]) -> str | None:
    normalized_labels = tuple(_normalize_text(label) for label in labels)
    for index, text in enumerate(texts):
        normalized_text = _normalize_text(text)
        for label in normalized_labels:
            if label not in normalized_text:
                continue
            if ":" in text:
                _, value = text.split(":", 1)
                value = value.strip()
                if value:
                    return value
            next_text = _next_meaningful_text(texts, index + 1)
            if next_text is not None:
                return next_text
    return None


def _extract_motivo(texts: list[str]) -> str | None:
    labeled_motivo = _extract_labeled_value(texts, _MOTIVO_LABELS)
    if labeled_motivo is not None:
        return labeled_motivo

    for index, text in enumerate(texts):
        normalized_text = _normalize_text(text)
        if "pago movil" in normalized_text or "transferencia" in normalized_text:
            next_text = _next_meaningful_text(texts, index + 1)
            if next_text and _normalize_text(next_text) not in {"banco destino", "comprobante"}:
                return next_text
    return None


def extract_receipt_data(image_bytes: bytes) -> OCRResult:
    return extract_receipt_snapshot(image_bytes).result


def extract_receipt_snapshot(image_bytes: bytes) -> OCRExtraction:
    """Full pipeline: raw image bytes → structured receipt data and OCR text."""
    img = _preprocess(image_bytes)
    prediction, _ = _get_engine()(img)
    texts = _clean_texts(prediction)

    result = OCRResult(
        valor=_extract_amount(texts),
        fecha=_extract_date(texts),
        banco_destino=_extract_labeled_value(texts, _BANK_LABELS),
        comprobante=_extract_labeled_value(texts, _RECEIPT_LABELS),
        motivo=_extract_motivo(texts),
    )
    return OCRExtraction(result=result, texts=texts)
