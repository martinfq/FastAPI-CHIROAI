import asyncio
import logging
from time import perf_counter

from app.models.ocr import ProcessReceiptResponse
from app.models.telegram import ReceiptMessage
from app.services import deepseek as deepseek_service
from app.services import ocr as ocr_service
from app.services.providers import get_provider_handler

logger = logging.getLogger(__name__)


async def process_receipt_message(message: ReceiptMessage) -> ProcessReceiptResponse:
    started_at = perf_counter()

    handler = get_provider_handler(message.provider)
    image_bytes = await handler.download_image(message)

    ocr_started_at = perf_counter()
    extraction = await asyncio.to_thread(ocr_service.extract_receipt_snapshot, image_bytes)
    ocr_elapsed_ms = (perf_counter() - ocr_started_at) * 1000
    logger.info(
        "OCR processing time provider=%s image_path=%s: %.2f ms",
        message.provider,
        message.image_path,
        ocr_elapsed_ms,
    )

    deepseek_started_at = perf_counter()
    try:
        classification = await asyncio.to_thread(
            deepseek_service.classify_receipt,
            message,
            extraction.result,
            extraction.texts,
        )
    except Exception as exc:
        logger.exception(
            "DeepSeek classification failed provider=%s image_path=%s: %s",
            message.provider,
            message.image_path,
            exc,
        )
        classification = deepseek_service.fallback_classification(message.categories)

    deepseek_elapsed_ms = (perf_counter() - deepseek_started_at) * 1000
    logger.info(
        "DeepSeek processing time provider=%s image_path=%s: %.2f ms",
        message.provider,
        message.image_path,
        deepseek_elapsed_ms,
    )
    logger.info(
        "DeepSeek summary provider=%s image_path=%s: %s",
        message.provider,
        message.image_path,
        classification.model_dump(),
    )

    elapsed_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "Processed receipt provider=%s image_path=%s in %.2f ms",
        message.provider,
        message.image_path,
        elapsed_ms,
    )
    logger.info("OCR result provider=%s image_path=%s: %s", message.provider, message.image_path, extraction.result)
    logger.info("Classification provider=%s image_path=%s: %s", message.provider, message.image_path, classification)

    return ProcessReceiptResponse(ocr=extraction.result, clasificacion=classification)
