from typing import Any

from pydantic import BaseModel


class TelegramUpdate(BaseModel):
    model_config = {"extra": "allow"}

    update_id: int
    message: dict[str, Any] | None = None
    edited_message: dict[str, Any] | None = None
    callback_query: dict[str, Any] | None = None
    inline_query: dict[str, Any] | None = None


class ReceiptMessage(BaseModel):
    type: str
    chat_id: int
    update_id: int
    message_id: int
    image_path: str
    file_id: str
