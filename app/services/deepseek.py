import json
import logging
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.ocr import ClassificationResult, OCRResult
from app.models.telegram import ReceiptCategory, ReceiptMessage, ReceiptSubcategory

logger = logging.getLogger(__name__)

_FALLBACK_CATEGORY = "Sin categoria"


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def fallback_classification(categories: list[ReceiptCategory]) -> ClassificationResult:
    fallback = _find_category(categories, _FALLBACK_CATEGORY)
    if fallback is None:
        return ClassificationResult(categoria_id=None, subcategoria_id=None)
    return ClassificationResult(categoria_id=fallback.id, subcategoria_id=None)


def build_classification_payload(
    message: ReceiptMessage,
    ocr_result: OCRResult,
    texts: list[str],
) -> dict[str, Any]:
    return {
        "categories": [
            {
                "id": category.id,
                "name": category.name,
                "subcategories": [
                    {"id": subcategory.id, "name": subcategory.name}
                    for subcategory in category.subcategories or []
                ],
            }
            for category in message.categories
        ],
        "ocr": {
            "valor": ocr_result.valor,
            "fecha": ocr_result.fecha.isoformat() if ocr_result.fecha is not None else None,
            "banco_destino": ocr_result.banco_destino,
            "comprobante": ocr_result.comprobante,
            "motivo": ocr_result.motivo,
            "texto_detectado": texts,
        },
    }


def classify_receipt(
    message: ReceiptMessage,
    ocr_result: OCRResult,
    texts: list[str],
) -> ClassificationResult:
    payload = build_classification_payload(message, ocr_result, texts)
    completion = _get_client().chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(payload)},
        ],
        stream=False,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"
    raw_classification = _parse_json_content(content)
    return _validate_classification(raw_classification, message.categories)


def _build_system_prompt() -> str:
    return (
        "Eres un clasificador de gastos. "
        "Debes elegir una categoria y subcategoria unicamente a partir del catalogo recibido. "
        "Responde siempre con JSON valido con esta forma exacta: "
        '{"categoria_id":<id entero o null>,"subcategoria_id":<id entero o null>}. '
        "No agregues explicaciones, texto adicional ni markdown. "
        "Usa el OCR estructurado y el texto_detectado para inferir la mejor opcion. "
        "Si la evidencia no alcanza, usa la categoria 'Sin categoria' si existe en el catalogo. "
        "Si la categoria elegida no define subcategorias, responde subcategoria_id=null. "
        "Nunca inventes ids de categorias o subcategorias fuera del catalogo."
    )


def _build_user_prompt(payload: dict[str, Any]) -> str:
    return (
        "Clasifica el pago usando el siguiente JSON. "
        "Da prioridad a motivo, valor, banco_destino, comprobante y texto_detectado.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_json_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek response is not a JSON object")
    return parsed


def _validate_classification(
    candidate: dict[str, Any],
    categories: list[ReceiptCategory],
) -> ClassificationResult:
    selected_category = _find_category_by_id(categories, candidate.get("categoria_id"))
    if selected_category is None:
        return fallback_classification(categories)

    selected_subcategory = candidate.get("subcategoria_id")
    if not selected_category.subcategories:
        return ClassificationResult(categoria_id=selected_category.id, subcategoria_id=None)

    matched_subcategory = _find_subcategory_by_id(selected_category.subcategories, selected_subcategory)
    if matched_subcategory is not None:
        return ClassificationResult(
            categoria_id=selected_category.id,
            subcategoria_id=matched_subcategory.id,
        )

    return fallback_classification(categories)


def _find_category(categories: list[ReceiptCategory], candidate_name: Any) -> ReceiptCategory | None:
    if not isinstance(candidate_name, str):
        return None
    for category in categories:
        if category.name.casefold() == candidate_name.casefold():
            return category
    return None


def _find_category_by_id(categories: list[ReceiptCategory], candidate_id: Any) -> ReceiptCategory | None:
    if not isinstance(candidate_id, int):
        return None
    for category in categories:
        if category.id == candidate_id:
            return category
    return None


def _find_subcategory_by_id(
    subcategories: list[ReceiptSubcategory] | None,
    candidate_id: Any,
) -> ReceiptSubcategory | None:
    if not isinstance(candidate_id, int) or subcategories is None:
        return None
    for subcategory in subcategories:
        if subcategory.id == candidate_id:
            return subcategory
    return None