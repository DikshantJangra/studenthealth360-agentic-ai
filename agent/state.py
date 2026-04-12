"""
AgentState — typed state dict shared across all LangGraph nodes.
Every node reads from / writes to this shared state.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """Shared state passed through the LangGraph workflow."""

    # ── Input (set at entry) ────────────────────────────────────────
    patient_data: dict
    """Raw patient input dict — Age, Heart_Rate, BP, Stress, etc."""

    # ── ML prediction (set by entry logic before graph starts) ─────
    risk_score: float
    """Probability of predicted risk class (0.0 – 1.0)."""

    risk_class: int
    """Predicted class index: 0 = Low, 1 = Moderate, 2 = High."""

    risk_label: str
    """Human-readable risk label: 'Low', 'Moderate', 'High'."""

    top_features: List[str]
    """Top 3 features driving the prediction."""

    # ── RAG retrieval (set by rag_retriever node) ──────────────────
    retrieved_guidelines: List[str]
    """Relevant guideline text chunks from ChromaDB."""

    guideline_sources: List[str]
    """Source filenames for each retrieved chunk (for citation)."""

    # ── LLM reasoning (set by risk_analyser node) ──────────────────
    risk_analysis: Optional[str]
    """LLM analysis of why the patient has this risk level."""

    # ── Final output (set by report_generator node) ────────────────
    health_report: Optional[str]
    """Final structured health report with all 4 mandatory sections."""

    # ── Error handling ─────────────────────────────────────────────
    error: Optional[str]
    """Error message if any node fails; triggers error_handler route."""
