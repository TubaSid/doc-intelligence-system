from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    retrieval_score: float
    steps_taken: int
    has_hallucination: bool
    sources: List[dict]

class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100)
    # Will handle file upload separately

class IngestResponse(BaseModel):
    doc_id: str
    status: str
    chunks_created: int
    chunks_stored: int

class HealthResponse(BaseModel):
    status: str
    vector_count: int
    model_loaded: bool