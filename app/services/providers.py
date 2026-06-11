import logging
from dataclasses import dataclass
from typing import Protocol

from app.models.telegram import ReceiptMessage
from app.services import telegram as telegram_service
from app.services import whatsapp as whatsapp_service

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base provider processing error."""


class ProviderNotConfiguredError(ProviderError):
    """Raised when the provider does not have a registered handler."""


class ProviderDownloadError(ProviderError):
    """Raised when downloading media from provider fails."""


class ProviderHandler(Protocol):
    provider: str

    async def download_image(self, message: ReceiptMessage) -> bytes:
        """Download and return image bytes for the receipt message."""


@dataclass(slots=True)
class TelegramProviderHandler:
    provider: str = "telegram"

    async def download_image(self, message: ReceiptMessage) -> bytes:
        try:
            return await telegram_service.download_file(message.image_path)
        except Exception as exc:
            logger.error("Telegram download failed for image_path=%s: %s", message.image_path, exc)
            raise ProviderDownloadError("Could not download file from Telegram") from exc


@dataclass(slots=True)
class WhatsAppProviderHandler:
    provider: str = "whatsapp"

    async def download_image(self, message: ReceiptMessage) -> bytes:
        try:
            return await whatsapp_service.download_file(message.image_path)
        except Exception as exc:
            logger.error("WhatsApp download failed for image_path=%s: %s", message.image_path, exc)
            raise ProviderDownloadError("Could not download file from WhatsApp") from exc


_PROVIDER_HANDLERS: dict[str, ProviderHandler] = {
    "telegram": TelegramProviderHandler(),
    "whatsapp": WhatsAppProviderHandler(),
}


def get_provider_handler(provider: str) -> ProviderHandler:
    handler = _PROVIDER_HANDLERS.get(provider)
    if handler is None:
        raise ProviderNotConfiguredError(f"Provider '{provider}' is not configured")
    return handler
