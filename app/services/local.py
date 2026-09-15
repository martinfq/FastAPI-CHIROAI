import asyncio
import logging
from time import perf_counter

from app.models.ocr import OCRResult
from app.services import ocr as ocr_service

logger = logging.getLogger(__name__)


async def process_receipt_image(image_bytes: bytes) -> OCRResult:
    """Run preprocessing and OCR directly on raw image bytes, no validation."""
    started_at = perf_counter()

    extraction = await asyncio.to_thread(ocr_service.extract_receipt_snapshot, image_bytes)

    elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info("Local OCR processing time: %.2f ms", elapsed_ms)
    logger.info("Local OCR result: %s", extraction.result)

    return extraction.result
