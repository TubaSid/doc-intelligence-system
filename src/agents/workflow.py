from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    router_node,
    retriever_node,
    answerer_node,
    verifier_node,
    fallback_node
)

def should_retrieve(state: dict) -> str:
    """Routing logic after retrieval"""
    if state.get("retrieval_score", 0) < 0.4:
        return "fallback"
    return "answer"

def should_fallback(state: dict) -> str:
    """Check if we need fallback after answering"""
    if state.get("answer_confidence", 0) < 0.4:
        return "fallback"
    if state.get("has_hallucination", False):
        return "fallback"
    return "end"

def create_workflow():
    """Create LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("answer", answerer_node)
    workflow.add_node("verify", verifier_node)
    workflow.add_node("fallback", fallback_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add edges
    workflow.add_edge("router", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        should_retrieve,
        {
            "answer": "answer",
            "fallback": "fallback"
        }
    )
    workflow.add_edge("answer", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_fallback,
        {
            "fallback": "fallback",
            "end": END
        }
    )
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

# Create compiled graph
agent_graph = create_workflow()