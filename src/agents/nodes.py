from typing import List
from langchain_groq import ChatGroq
from groq import Groq
import os
from dotenv import load_dotenv
from vector_store import get_embedding, pc, INDEX_NAME

load_dotenv()

# Initialize clients
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Smaller, more efficient model
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
index = pc.Index(INDEX_NAME)

# ===== NODE 1: ROUTER =====
def router_node(state: dict) -> dict:
    """Decides if query needs retrieval or can answer directly"""
    query = state["query"]
    
    prompt = f"""Analyze this query: "{query}"
    
    Does this query require searching documents, or can it be answered with general knowledge?
    
    Respond with only: SEARCH or GENERAL"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    decision = response.choices[0].message.content.strip()
    
    print(f"Router: {decision}")
    
    state["step_count"] = 1
    return state

# ===== NODE 2: RETRIEVER =====
def retriever_node(state: dict) -> dict:
    """Retrieves relevant chunks from vector DB"""
    query = state["query"]
    
    print(f"Retrieving chunks for: {query}")
    
    # Get embedding
    query_embedding = get_embedding(query)
    
    # Search with higher top_k for better coverage
    results = index.query(
        vector=query_embedding,
        top_k=10,  # Increased from 5
        include_metadata=True
    )
    
    # Calculate average score
    avg_score = sum(m.score for m in results.matches) / len(results.matches) if results.matches else 0
    
    # Store chunks
    state["retrieved_chunks"] = [
        {
            "text": m.metadata["text"],
            "score": m.score,
            "doc_id": m.metadata["doc_id"],
            "chunk_id": m.metadata["chunk_id"]
        }
        for m in results.matches
    ]
    state["retrieval_score"] = avg_score
    state["step_count"] += 1
    
    print(f"Retrieved {len(results.matches)} chunks (avg score: {avg_score:.4f})")
    
    return state

# ===== NODE 3: ANSWERER =====
def answerer_node(state: dict) -> dict:
    """Generates answer from retrieved context"""
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    print(f"Generating answer...")
    
    # Build context
    context = "\n\n".join([
        f"[Source {i+1}] (Score: {c['score']:.3f}):\n{c['text']}"
        for i, c in enumerate(chunks[:5])  # Top 5 only
    ])
    
    prompt = f"""You are a financial document analyst. Answer the question using ONLY the provided sources.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the context
- Cite source numbers [Source X]
- If information is not in context, say "I cannot find this information"
- Be specific with numbers and facts

Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    
    # Simple confidence scoring
    if "cannot find" in answer.lower() or "not in" in answer.lower():
        confidence = 0.3
    elif state["retrieval_score"] > 0.6:
        confidence = 0.8
    else:
        confidence = 0.5
    
    state["answer"] = answer
    state["answer_confidence"] = confidence
    state["step_count"] += 1
    
    print(f"Answer generated (confidence: {confidence:.2f})")
    
    return state

# ===== NODE 4: VERIFIER =====
def verifier_node(state: dict) -> dict:
    """Checks for hallucinations by comparing answer to sources"""
    answer = state["answer"]
    chunks = state["retrieved_chunks"]
    
    print(f"Verifying answer...")
    
    # Build source text
    source_text = "\n".join([c["text"] for c in chunks[:5]])
    
    prompt = f"""Compare the answer to the source documents. Check if the answer contains information NOT present in the sources.

Sources:
{source_text}

Answer:
{answer}

Does the answer contain hallucinated information (facts not in sources)?
Respond with: YES or NO, followed by brief explanation."""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    verification = response.choices[0].message.content
    
    has_hallucination = "YES" in verification.split("\n")[0].upper()
    
    state["has_hallucination"] = has_hallucination
    state["verification_notes"] = verification
    state["step_count"] += 1
    
    status = "HALLUCINATION DETECTED" if has_hallucination else "Verified"
    print(f"{status}")
    
    return state

# ===== NODE 5: FALLBACK =====
def fallback_node(state: dict) -> dict:
    """Handles low confidence or failed retrievals"""
    print(f"Fallback triggered")
    
    state["answer"] = f"""I couldn't find reliable information to answer: "{state['query']}"

This could be because:
- The information isn't in the indexed documents
- The query needs rephrasing
- More context is needed

Retrieval score: {state.get('retrieval_score', 0):.3f}
Confidence: {state.get('answer_confidence', 0):.3f}

Suggestions:
- Try rephrasing your question
- Check if this information is in the document
- Provide more specific details"""
    
    state["step_count"] += 1
    return state