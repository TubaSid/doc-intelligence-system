from agents.workflow import agent_graph
from agents.state import AgentState

def ask_agent(query: str):
    """Query using agent workflow"""
    
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")
    
    # Initialize state
    initial_state = AgentState(
        query=query,
        retrieved_chunks=[],
        retrieval_score=0.0,
        answer="",
        answer_confidence=0.0,
        has_hallucination=False,
        verification_notes="",
        step_count=0,
        error=""
    )
    
    # Run workflow
    result = agent_graph.invoke(initial_state)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"ANSWER:")
    print(f"{'='*70}")
    print(result["answer"])
    
    print(f"\n{'='*70}")
    print(f"METADATA:")
    print(f"{'='*70}")
    print(f"Steps taken: {result['step_count']}")
    print(f"Retrieval score: {result.get('retrieval_score', 0):.4f}")
    print(f"Answer confidence: {result.get('answer_confidence', 0):.4f}")
    print(f"Hallucination check: {'PASS' if not result.get('has_hallucination') else 'FAIL'}")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    queries = [
        "What was Tesla's total revenue in 2023?",
        "How many vehicles did Tesla deliver?",
        "What are the main risk factors?"
    ]
    
    for query in queries:
        ask_agent(query)
        input("Press Enter to continue...")