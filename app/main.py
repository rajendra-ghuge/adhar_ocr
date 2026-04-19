from fastapi import FastAPI
from app.inference import router

app = FastAPI(title="YOLO OCR API")

app.include_router(router)