from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.routes import router
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="Production RAG system with multi-agent workflow",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Document Intelligence"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Document Intelligence API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,  # Pass app object directly
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload for stability
    )