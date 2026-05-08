import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import telegram
from app.services import ocr as ocr_service

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Warming up OCR engine...")
    ocr_service.warmup()
    logging.info("OCR engine ready.")
    yield
    logging.info("Shutting down...")


app = FastAPI(title="ia-money", lifespan=lifespan)

app.include_router(telegram.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
