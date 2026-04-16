"""
AgentState — typed state dict shared across all LangGraph nodes.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """Shared state passed through the LangGraph workflow."""

    # ── Student Profile (set at entry) ─────────────────────────────
    user_name: str
    primary_goal: str
    journal_entry: str

    # ── Input (set at entry) ────────────────────────────────────────
    patient_data: dict
    """Raw patient input dict — Age, Heart_Rate, BP, Stress, etc."""

    # ── ML prediction (set by entry logic before graph starts) ─────
    risk_score: float
    """Probability of predicted risk class (0.0 – 1.0)."""

    risk_label: str
    """Human-readable risk label: 'Low', 'Moderate', 'High'."""

    top_features: List[str]
    """Top 3 features driving the prediction."""

    # ── RAG retrieval (set by rag_retriever node) ──────────────────
    retrieved_guidelines: List[str]
    """Relevant guideline text chunks from ChromaDB."""

    guideline_sources: List[str]
    """Source filenames for each retrieved chunk."""

    # ── LLM reasoning (set by risk_analyser node) ──────────────────
    risk_analysis: Optional[str]
    """Detailed clinical reasoning analysis."""

    # ── Final output (set by report_generator node) ────────────────
    health_report: Optional[str]
    """Final structured health report with all 4 mandatory sections."""

    # ── Error handling ─────────────────────────────────────────────
    error: Optional[str]
    """Error message if any node fails; triggers error_handler route."""
