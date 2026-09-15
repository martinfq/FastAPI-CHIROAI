from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UserSubcategory(BaseModel):
    id: int | None = None
    name: str


class UserCategory(BaseModel):
    id: int | None = None
    name: str
    subcategories: list[UserSubcategory] | None = None

class RequestLocal(BaseModel):
    model_config = ConfigDict(extra="forbid")

class RequestOCR(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: list[UserCategory]

class ResquestOCRWhatsapp(RequestOCR):

    provider: Literal["whatsapp"] = Field(
        description="Origin provider for the receipt payload."
    )
    type: Literal["image"]
    image_path: str
    chat_id: int | None = None
    update_id: int | None = None
    message_id: int | None = None
    file_id: str | None = None