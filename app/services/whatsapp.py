import httpx

from app.config import settings


async def download_file(file_url: str) -> bytes:
    headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, headers=headers)
    response.raise_for_status()
    return response.content
