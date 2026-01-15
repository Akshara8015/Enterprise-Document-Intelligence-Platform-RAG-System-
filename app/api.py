import time
import logging
from fastapi import APIRouter, HTTPException
from rag.runtime_index import build_vectorstore_from_pdfs

from app.schemas import QueryRequest, QueryResponse
from rag.generator import rag_answer
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_rag_answer(query:str , vectorstore):
    return rag_answer(query, vectorstore)


router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):

    start = time.time()

    if not request.pdf_paths or len(request.pdf_paths) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least one PDF is required"
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    vectorstore = build_vectorstore_from_pdfs(request.pdf_paths)
    result = cached_rag_answer(
        query=request.question,
        vectorstore = vectorstore
    )

    result["time_taken"] = round(time.time() - start, 2)
    return result

