"""
LangGraph workflow definition for the StudentHealth360 Agentic AI.

Graph topology (Parallel Fan-out):
    START ──┬──▶ risk_analyser ──┐
            │                     ├──▶ report_generator ──▶ END
            └──▶ rag_retriever ──┘         ▲
                                           │
                                     (error_handler)
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.risk_analyser import risk_analyser_node
from agent.nodes.rag_retriever import rag_retriever_node
from agent.nodes.report_generator import report_generator_node
from agent.nodes.error_handler import error_handler_node


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state graph with parallel fan-out.
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    workflow.add_node("risk_analyser", risk_analyser_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("error_handler", error_handler_node)

    # ── Entry point and Parallel Fan-out ────────────────────────────
    # Entry node that branches to both parallel tasks
    def start_node(state: AgentState): return state
    workflow.add_node("start_node", start_node)
    workflow.set_entry_point("start_node")
    
    workflow.add_edge("start_node", "risk_analyser")
    workflow.add_edge("start_node", "rag_retriever")
    
    # Fan-in to report generator
    workflow.add_edge("risk_analyser", "report_generator")
    workflow.add_edge("rag_retriever", "report_generator")
    
    # Final transition
    workflow.add_edge("report_generator", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


# Pre-compiled graph instance
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    user_name: str,
    primary_goal: str,
    journal_entry: str,
    patient_data: dict,
    risk_score: float,
    risk_label: str,
    top_features: list,
) -> AgentState:
    """
    Execute the full agent workflow with personalized student context.
    """
    initial_state: AgentState = {
        "user_name": user_name,
        "primary_goal": primary_goal,
        "journal_entry": journal_entry,
        "patient_data": patient_data,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "top_features": top_features,
        "retrieved_guidelines": [],
        "risk_analysis": None,
        "health_report": None,
        "error": None,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
