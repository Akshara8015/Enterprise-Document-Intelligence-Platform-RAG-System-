# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    pdf_paths: List[str]


class SourceInfo(BaseModel):
    document: str
    section: Optional[str] = None
    pages: Optional[str] = None
    chunk_id: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence_score: Optional[float]
    is_grounded: Optional[bool]
    num_sources: int
    time_taken: float
    sources: List[SourceInfo]
