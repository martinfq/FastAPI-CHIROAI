from typing import Any, Literal

from pydantic import BaseModel


class ReceiptSubcategory(BaseModel):
    id: int | None = None
    name: str


class ReceiptCategory(BaseModel):
    id: int | None = None
    name: str
    subcategories: list[ReceiptSubcategory] | None = None


class TelegramUpdate(BaseModel):
    model_config = {"extra": "allow"}

    update_id: int
    message: dict[str, Any] | None = None
    edited_message: dict[str, Any] | None = None
    callback_query: dict[str, Any] | None = None
    inline_query: dict[str, Any] | None = None


class ReceiptMessage(BaseModel):
    type: Literal["image"]
    chat_id: int
    update_id: int
    message_id: int
    categories: list[ReceiptCategory]
    image_path: str
    file_id: str
