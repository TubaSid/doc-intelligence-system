import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load free local embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions, fast and free
print("✓ Embedding model loaded")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "doc-intelligence"

def initialize_index():
    """Create Pinecone index if it doesn't exist"""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✓ Created index: {INDEX_NAME}")
    else:
        print(f"✓ Index already exists: {INDEX_NAME}")
    
    return pc.Index(INDEX_NAME)

def get_embedding(text: str) -> List[float]:
    """Get embedding from local model"""
    embedding = embedding_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()

def upsert_chunks(chunks: List[Dict], doc_id: str, index):
    """Store chunks in Pinecone with embeddings"""
    vectors = []
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        # Get embedding
        embedding = get_embedding(chunk["text"])
        
        # Prepare vector
        vectors.append({
            "id": f"{doc_id}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "doc_id": doc_id,
                "chunk_id": i,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"]
            }
        })
        
        # Show progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")
    
    # Upsert to Pinecone in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"✓ Uploaded batch {i//batch_size + 1}")
    
    print(f"✓ Stored {len(vectors)} vectors for {doc_id}")

if __name__ == "__main__":
    # Test connection
    index = initialize_index()
    print(f"✓ Connected to index: {INDEX_NAME}")
    
    # Check stats
    stats = index.describe_index_stats()
    print(f"✓ Index has {stats['total_vector_count']} vectors")
