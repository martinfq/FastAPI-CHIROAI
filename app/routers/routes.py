import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.ocr import OCRResult, ProcessReceiptResponse
from app.models.telegram import ReceiptMessage
from app.services.local import process_receipt_image
from app.services.receipt_processor import process_receipt_message
from app.services.providers import (
    ProviderDownloadError,
    ProviderNotConfiguredError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/iaservice", tags=["iaservice"])


@router.post("/process-image", response_model=ProcessReceiptResponse)
async def process_receipt(body: ReceiptMessage) -> ProcessReceiptResponse:
    """Process a receipt image based on provider."""
    try:
        return await process_receipt_message(body)
    except ProviderNotConfiguredError as exc:
        logger.error("Provider not configured provider=%s", body.provider)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderDownloadError as exc:
        logger.error("Provider download failed provider=%s image_path=%s", body.provider, body.image_path)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected processing error provider=%s image_path=%s", body.provider, body.image_path)
        raise HTTPException(status_code=500, detail="Internal error while processing receipt") from exc

@router.post("/process-image-local", response_model=OCRResult)
async def process_receipt_local(image: UploadFile = File(...)) -> OCRResult:
    """Process a receipt image sent as multipart/form-data, no validation, OCR only."""
    image_bytes = await image.read()
    return await process_receipt_image(image_bytes)
