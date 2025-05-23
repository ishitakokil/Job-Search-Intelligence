from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict, List, Literal
from agents import (
    query_rewriter,
    retriever_agent,
    grade_documents,
    verification_agent,
    generate_agent,
    generate_agent_lora,
)

class RAGState(TypedDict):
    query: str
    refined_query: str
    retrieved_docs: List[str]
    formatted_context: str
    llm_answer: str
    decision: Literal["relevant", "not_relevant", "useful", "not_useful", "end"]
    retries: int

def create_workflow(use_lora: bool = False):
    workflow = StateGraph(RAGState)

    summarizer = generate_agent_lora if use_lora else generate_agent

    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("retriever_agent", retriever_agent)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_agent", summarizer)
    workflow.add_node("verification_agent", verification_agent)

    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "retriever_agent")
    workflow.add_edge("retriever_agent", "grade_documents")
    workflow.add_conditional_edges("grade_documents", lambda state: state["decision"], {
        "relevant": "generate_agent",
        "not_relevant": "query_rewriter",
        "end": END
    })
    workflow.add_edge("generate_agent", "verification_agent")
    workflow.add_conditional_edges("verification_agent", lambda state: state["decision"], {
        "useful": END,
        "not_useful": "query_rewriter",
        "end": END
    })

    return workflow.compile()

