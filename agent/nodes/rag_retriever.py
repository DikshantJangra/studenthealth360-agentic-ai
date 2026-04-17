"""
RAG Retriever Node — Queries ChromaDB for relevant medical guidelines.
"""

from agent.state import AgentState
from agent.factory import get_vectorstore
from config import RAG_TOP_K


def _build_search_query(state: AgentState) -> str:
    """
    Construct a localized semantic search query from the risk profile and student goals.
    """
    risk_label = state.get("risk_label", "moderate")
    top_features = state.get("top_features", [])
    primary_goal = state.get("primary_goal", "")
    journal_entry = state.get("journal_entry", "")

    # Build a context-rich query for ChromaDB
    parts = [f"{risk_label} health risk guidelines"]
    if primary_goal:
        parts.append(f"focus on {primary_goal}")
    if journal_entry:
        # Use a snippet of the journal to avoid query bloat
        parts.append(f"student context: {journal_entry[:100]}")
    if top_features:
        parts.append("metrics: " + ", ".join(top_features))

    return " ".join(parts)


def rag_retriever_node(state: AgentState) -> dict:
    """
    Retrieve relevant medical guideline chunks from ChromaDB with citations.
    """
    try:
        vectorstore = get_vectorstore()
        query = _build_search_query(state)
        results = vectorstore.similarity_search(query, k=RAG_TOP_K)

        guidelines = [doc.page_content for doc in results]
        sources = [doc.metadata.get("source", "Unknown Guideline") for doc in results]

        return {
            "retrieved_guidelines": guidelines,
            "guideline_sources": sources
        }

    except Exception as e:
        return {"error": f"RAG retrieval failed: {str(e)}"}
