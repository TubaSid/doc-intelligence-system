from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    """State passed between agents"""
    # Input
    query: str
    
    # Retrieved context
    retrieved_chunks: List[dict]
    retrieval_score: float
    
    # Generated answer
    answer: str
    answer_confidence: float
    
    # Verification
    has_hallucination: bool
    verification_notes: str
    
    # Metadata
    step_count: Annotated[int, operator.add]
    error: str