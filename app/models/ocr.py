from datetime import date

from pydantic import BaseModel


class OCRResult(BaseModel):
    valor: float | None = None
    fecha: date | None = None
    banco_destino: str | None = None
    comprobante: str | None = None
