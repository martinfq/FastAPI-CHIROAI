import logging

from fastapi import APIRouter, HTTPException

from app.models.ocr import OCRResult
from app.models.telegram import ReceiptMessage
from app.services import ocr as ocr_service
from app.services import telegram as telegram_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/process-receipt", response_model=OCRResult)
async def process_receipt(body: ReceiptMessage) -> OCRResult:
    """Process a receipt image from Telegram."""
    try:
        image_bytes = await telegram_service.download_file(body.image_path)
    except Exception as exc:
        logger.error("Telegram download failed for image_path=%s: %s", body.image_path, exc)
        raise HTTPException(status_code=502, detail="Could not download file from Telegram") from exc

    result = ocr_service.extract_receipt_data(image_bytes)
    logger.info("OCR result for image_path=%s: %s", body.image_path, result)
    return result
