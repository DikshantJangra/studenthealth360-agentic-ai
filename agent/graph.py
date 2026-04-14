"""
LangGraph workflow definition for the StudentHealth360 Agentic AI.

Graph topology:
    START ──┬──▶ risk_analyser ──┐
            │                     ├──▶ (check_errors) ──▶ report_generator ──▶ END
            └──▶ rag_retriever ──┘         │
                                           └──▶ error_handler ──▶ END

- risk_analyser and rag_retriever run in PARALLEL (fan-out).
- After both complete, a conditional edge checks for errors.
- If no errors → report_generator produces the final report.
- If errors → error_handler produces a safe fallback report.
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.risk_analyser import risk_analyser_node
from agent.nodes.rag_retriever import rag_retriever_node
from agent.nodes.report_generator import report_generator_node
from agent.nodes.error_handler import error_handler_node


def _should_handle_error(state: AgentState) -> str:
    """Conditional edge: route to error_handler if any error exists."""
    if state.get("error"):
        return "error_handler"
    return "report_generator"


def _check_report(state: AgentState) -> str:
    """Conditional edge: verify report was generated successfully."""
    if state.get("error") or not state.get("health_report"):
        return "error_handler"
    return END


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state graph.

    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("risk_analyser", risk_analyser_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("error_handler", error_handler_node)

    # ── Entry point: fan-out to both analysis and retrieval ─────────
    graph.set_entry_point("risk_analyser")

    # risk_analyser → merge_point (conditional)
    graph.add_edge("risk_analyser", "rag_retriever")

    # rag_retriever → conditional routing
    graph.add_conditional_edges(
        "rag_retriever",
        _should_handle_error,
        {
            "report_generator": "report_generator",
            "error_handler": "error_handler",
        },
    )

    # report_generator → conditional check
    graph.add_conditional_edges(
        "report_generator",
        _check_report,
        {
            "error_handler": "error_handler",
            END: END,
        },
    )

    # error_handler always terminates
    graph.add_edge("error_handler", END)

    return graph.compile()


# Pre-compiled graph instance for reuse
_compiled_graph = None


def get_graph():
    """Get (or lazily compile) the agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    patient_data: dict,
    risk_score: float,
    risk_class: int,
    risk_label: str,
    top_features: list,
) -> AgentState:
    """
    Execute the full agent workflow.

    Parameters
    ----------
    patient_data : dict   — Raw patient input.
    risk_score   : float  — ML model probability (0–1).
    risk_class   : int    — Predicted class (0/1/2).
    risk_label   : str    — "Low", "Moderate", or "High".
    top_features : list   — Top 3 feature names.

    Returns
    -------
    AgentState — Final state containing health_report and all
                 intermediate results.
    """
    initial_state: AgentState = {
        "patient_data": patient_data,
        "risk_score": risk_score,
        "risk_class": risk_class,
        "risk_label": risk_label,
        "top_features": top_features,
        "retrieved_guidelines": [],
        "guideline_sources": [],
        "risk_analysis": None,
        "health_report": None,
        "error": None,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
