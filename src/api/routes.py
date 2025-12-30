from fastapi import APIRouter, HTTPException, UploadFile, File
from api.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, HealthResponse
from agents.workflow import agent_graph
from agents.state import AgentState
from ingest import extract_text_from_pdf, chunk_text
from vector_store import initialize_index, upsert_chunks, pc, INDEX_NAME
from pathlib import Path
import tempfile
import shutil

router = APIRouter()

# ===== QUERY ENDPOINT =====
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about indexed documents"""
    
    try:
        # Initialize state
        initial_state = AgentState(
            query=request.query,
            retrieved_chunks=[],
            retrieval_score=0.0,
            answer="",
            answer_confidence=0.0,
            has_hallucination=False,
            verification_notes="",
            step_count=0,
            error=""
        )
        
        # Run agent workflow
        result = agent_graph.invoke(initial_state)
        
        # Format response
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            confidence=result.get("answer_confidence", 0.0),
            retrieval_score=result.get("retrieval_score", 0.0),
            steps_taken=result["step_count"],
            has_hallucination=result.get("has_hallucination", False),
            sources=[
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                    "text_preview": chunk["text"][:200]
                }
                for chunk in result.get("retrieved_chunks", [])[:5]
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ===== INGEST ENDPOINT =====
@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(doc_id: str, file: UploadFile = File(...)):
    """Upload and index a new document"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Extract text
        text = extract_text_from_pdf(tmp_path)
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        # Initialize index
        index = initialize_index()
        
        # Store chunks
        upsert_chunks(chunks, doc_id, index)
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return IngestResponse(
            doc_id=doc_id,
            status="success",
            chunks_created=len(chunks),
            chunks_stored=len(chunks)
        )
    
    except Exception as e:
        # Cleanup on error
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# ===== HEALTH ENDPOINT =====
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    
    try:
        # Check Pinecone connection
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        
        return HealthResponse(
            status="healthy",
            vector_count=stats.get('total_vector_count', 0),
            model_loaded=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")