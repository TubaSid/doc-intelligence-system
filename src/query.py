import os
from dotenv import load_dotenv
from vector_store import get_embedding, pc, INDEX_NAME
from groq import Groq

load_dotenv()

# Initialize Groq for answer generation (free and fast)
client = Groq(api_key=os.getenv("XAI_API_KEY"))
index = pc.Index(INDEX_NAME)

def search(query: str, top_k: int = 5):
    """Search vector database for relevant chunks"""
    print(f"\nSearching for: '{query}'")
    
    # Get query embedding using same model
    query_embedding = get_embedding(query)
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    print(f"✓ Found {len(results.matches)} results")
    
    return results.matches

def generate_answer(query: str, context_chunks: list) -> str:
    """Generate answer using GPT with retrieved context"""
    
    # Build context from top chunks
    context = "\n\n---\n\n".join([
        f"[Source {i+1}]: {match.metadata['text']}" 
        for i, match in enumerate(context_chunks)
    ])
    
    # Create prompt
    messages = [
        {
            "role": "system", 
            "content": "You are a financial document analyst. Answer questions based ONLY on the provided context. If the answer is not in the context, say 'I cannot find this information in the document.' Always cite which source number you're using."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    
    # Generate answer
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # Smaller, more efficient model
    messages=messages,
    temperature=0.3
)
    
    return response.choices[0].message.content

def ask(query: str):
    """Full RAG pipeline: search + generate"""
    
    print(f"\n{'='*70}")
    print(f"QUESTION: {query}")
    print(f"{'='*70}")
    
    # Step 1: Retrieve relevant chunks
    chunks = search(query, top_k=5)
    
    if not chunks:
        print("\n❌ No relevant information found")
        return
    
    # Step 2: Generate answer
    print("\n🤖 Generating answer...")
    answer = generate_answer(query, chunks)
    
    print(f"\n{'='*70}")
    print(f"ANSWER:")
    print(f"{'='*70}")
    print(answer)
    
    # Show sources
    print(f"\n{'='*70}")
    print(f"SOURCES:")
    print(f"{'='*70}")
    for i, match in enumerate(chunks):
        print(f"\n[{i+1}] Score: {match.score:.4f}")
        print(f"Doc: {match.metadata['doc_id']}, Chunk: {match.metadata['chunk_id']}")
        print(f"Text preview: {match.metadata['text'][:200]}...")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    # Test queries
    queries = [
        "What was Tesla's total revenue in 2023?",
        "How many vehicles did Tesla deliver?",
        "What are Tesla's main risk factors?"
    ]
    
    for query in queries:
        ask(query)
        input("\nPress Enter for next question...")