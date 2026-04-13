"""
RAG Retriever Node — Queries ChromaDB for relevant medical guidelines.
Uses semantic search based on risk level and top contributing features.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from agent.state import AgentState
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RAG_TOP_K,
)


def _build_search_query(state: AgentState) -> str:
    """
    Construct a semantic search query from the risk profile.
    Combines risk label with top features for targeted retrieval.
    """
    risk_label = state.get("risk_label", "moderate")
    top_features = state.get("top_features", [])
    patient = state.get("patient_data", {})

    # Build a natural-language query for semantic similarity
    parts = [f"{risk_label} health risk"]

    # Add feature-specific context
    feature_keywords = {
        "Heart Rate (BPM)": "cardiovascular heart rate monitoring",
        "Systolic Blood Pressure": "blood pressure hypertension",
        "Diastolic Blood Pressure": "blood pressure cardiovascular",
        "Biosensor Stress Level": "stress management chronic stress",
        "Self-Reported Stress Level": "stress anxiety mental health",
        "Weekly Study Hours": "academic burnout workload",
        "Weekly Project Hours": "academic burnout study hours",
        "Physical Activity Level": "physical activity exercise sedentary",
        "Sleep Quality": "sleep quality insomnia sleep hygiene",
        "Mood (Stressed)": "mental health stress mood disorders",
        "Mood (Happy)": "mental health well-being",
        "Mood (Neutral)": "emotional assessment mood monitoring",
    }

    for feat in top_features:
        if feat in feature_keywords:
            parts.append(feature_keywords[feat])

    # Add patient-specific context
    sleep = patient.get("Sleep_Quality", "")
    if sleep == "Poor":
        parts.append("poor sleep deprivation")
    activity = patient.get("Physical_Activity", "")
    if activity == "Low":
        parts.append("sedentary lifestyle low activity")
    mood = patient.get("Mood", "")
    if mood == "Stressed":
        parts.append("stressed anxious burnout")

    return " ".join(parts)


def rag_retriever_node(state: AgentState) -> dict:
    """
    Retrieve relevant medical guideline chunks from ChromaDB.

    Reads: risk_label, top_features, patient_data
    Writes: retrieved_guidelines, guideline_sources (or error)
    """
    try:
        # Load the embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )

        # Connect to persisted ChromaDB
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )

        # Build and execute semantic query
        query = _build_search_query(state)
        results = vectorstore.similarity_search(query, k=RAG_TOP_K)

        if not results:
            return {
                "retrieved_guidelines": [],
                "guideline_sources": [],
                "error": "No relevant guidelines found in the knowledge base.",
            }

        # Extract text and source metadata
        guidelines = []
        sources = set()
        for doc in results:
            guidelines.append(doc.page_content)
            source = doc.metadata.get("source", "Unknown")
            sources.add(source)

        # Deduplicate sources and format for display
        source_display = {
            "who_ncd_prevention.txt": "WHO — Global Action Plan for NCD Prevention",
            "cdc_sleep_guidelines.txt": "CDC — Sleep and Sleep Disorders Guidelines",
            "aha_cardiovascular_risk.txt": "AHA — Cardiovascular Risk Reduction Guidelines",
            "nih_stress_management.txt": "NIH — Stress Management and Mental Health",
            "acsm_physical_activity.txt": "ACSM — Physical Activity Guidelines",
            "apa_student_mental_health.txt": "APA — Student Mental Health and Burnout",
        }

        formatted_sources = [
            source_display.get(s, s) for s in sorted(sources)
        ]

        return {
            "retrieved_guidelines": guidelines,
            "guideline_sources": formatted_sources,
        }

    except Exception as e:
        return {
            "retrieved_guidelines": [],
            "guideline_sources": [],
            "error": f"RAG retrieval failed: {str(e)}",
        }
