from ingest import extract_text_from_pdf, chunk_text
from vector_store import initialize_index, upsert_chunks
from pathlib import Path

def process_document(pdf_path: str, doc_id: str):
    """Full pipeline: extract → chunk → embed → store"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract text
    print("Step 1: Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    # Step 2: Chunk text
    print("\nStep 2: Chunking text...")
    # Optimized chunking: larger chunks keep related info together
    chunks = chunk_text(text, chunk_size=2000, overlap=400)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Initialize Pinecone
    print("\nStep 3: Connecting to Pinecone...")
    index = initialize_index()
    
    # Step 4: Embed and store
    print("\nStep 4: Creating embeddings and storing...")
    upsert_chunks(chunks, doc_id, index)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {doc_id} indexed successfully")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Process your Tesla 10-K
    pdf_path = Path("d:/learn/End-to-end proj/data/raw/tesla_10k.pdf")
    process_document(
        pdf_path=str(pdf_path),
        doc_id="tesla_10k_2023"
    )
