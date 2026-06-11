from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="forbid")

    provider: Literal["telegram", "whatsapp"] = Field(
        description="Origin provider for the receipt payload."
    )
    type: Literal["image"]
    categories: list[ReceiptCategory]
    image_path: str
    chat_id: int | None = None
    update_id: int | None = None
    message_id: int | None = None
    file_id: str | None = None
