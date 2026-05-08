import httpx

from app.config import settings

_TELEGRAM_BASE_URL = "https://api.telegram.org"
_TELEGRAM_BOT_API = f"{_TELEGRAM_BASE_URL}/bot{{token}}"
_TELEGRAM_FILE_API = f"{_TELEGRAM_BASE_URL}/file/bot{{token}}/{{file_path}}"


async def get_file_path(file_id: str) -> str:
    url = f"{_TELEGRAM_BASE_URL}/bot{settings.telegram_token}/getFile"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params={"file_id": file_id})
    response.raise_for_status()
    return response.json()["result"]["file_path"]


async def download_file(file_path: str) -> bytes:
    url = f"{_TELEGRAM_BASE_URL}/file/bot{settings.telegram_token}/{file_path}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()
    return response.content
