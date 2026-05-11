import asyncio
import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException

from app.models.ocr import ProcessReceiptResponse
from app.models.telegram import ReceiptMessage
from app.services import deepseek as deepseek_service
from app.services import ocr as ocr_service
from app.services import telegram as telegram_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/process-receipt", response_model=ProcessReceiptResponse)
async def process_receipt(body: ReceiptMessage) -> ProcessReceiptResponse:
    """Process a receipt image from Telegram."""
    started_at = perf_counter()
    try:
        image_bytes = await telegram_service.download_file(body.image_path)
    except Exception as exc:
        logger.error("Telegram download failed for image_path=%s: %s", body.image_path, exc)
        raise HTTPException(status_code=502, detail="Could not download file from Telegram") from exc

    ocr_started_at = perf_counter()
    extraction = await asyncio.to_thread(ocr_service.extract_receipt_snapshot, image_bytes)
    ocr_elapsed_ms = (perf_counter() - ocr_started_at) * 1000
    logger.info("OCR processing time for image_path=%s: %.2f ms", body.image_path, ocr_elapsed_ms)

    deepseek_started_at = perf_counter()
    try:
        classification = await asyncio.to_thread(
            deepseek_service.classify_receipt,
            body,
            extraction.result,
            extraction.texts,
        )
    except Exception as exc:
        logger.exception("DeepSeek classification failed for image_path=%s: %s", body.image_path, exc)
        classification = deepseek_service.fallback_classification(body.categories)
    deepseek_elapsed_ms = (perf_counter() - deepseek_started_at) * 1000
    logger.info("DeepSeek processing time for image_path=%s: %.2f ms", body.image_path, deepseek_elapsed_ms)
    logger.info("DeepSeek summary for image_path=%s: %s", body.image_path, classification.model_dump())

    elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info("Processed receipt image_path=%s in %.2f ms", body.image_path, elapsed_ms)
    logger.info("OCR result for image_path=%s: %s", body.image_path, extraction.result)
    logger.info("Classification for image_path=%s: %s", body.image_path, classification)
    return ProcessReceiptResponse(ocr=extraction.result, clasificacion=classification)
