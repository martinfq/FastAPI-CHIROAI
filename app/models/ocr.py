from datetime import date

from pydantic import BaseModel


class OCRResult(BaseModel):
    valor: float | None = None
    fecha: date | None = None
    banco_destino: str | None = None
    comprobante: str | None = None
    motivo: str | None = None


class ClassificationResult(BaseModel):
    categoria_id: int | None = None
    subcategoria_id: int | None = None


class ProcessReceiptResponse(BaseModel):
    ocr: OCRResult
    clasificacion: ClassificationResult
