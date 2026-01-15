# app/main.py
import logging
from fastapi import FastAPI
from app.api import router as query_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Enterprise Document Intelligence API",
    version="1.0"
)

app.include_router(query_router)

@app.get("/")
def health_check():
    return {"status": "ok"}


