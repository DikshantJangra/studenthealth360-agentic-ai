"""
Centralised configuration for StudentHealth360 Agentic AI.
Loads environment variables and exposes constants used across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── API / LLM ────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ── RAG / Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = str(BASE_DIR / "rag" / "vectorstore")
CHROMA_COLLECTION_NAME = "medical_guidelines"
RAG_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── ML Model paths ──────────────────────────────────────────────────
MODEL_PATH = str(BASE_DIR / "ml" / "model.pkl")
SCALER_PATH = str(BASE_DIR / "ml" / "scaler.pkl")

# ── Feature definitions (must match M1 training schema exactly) ─────
NUMERICAL_FEATURES = [
    "Heart_Rate",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Stress_Level_Biosensor",
    "Stress_Level_Self_Report",
    "Study_Hours",
    "Project_Hours",
]

MODEL_FEATURE_ORDER = [
    "Age",
    "Heart_Rate",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Stress_Level_Biosensor",
    "Stress_Level_Self_Report",
    "Physical_Activity",
    "Sleep_Quality",
    "Study_Hours",
    "Project_Hours",
    "Gender_F",
    "Gender_M",
    "Mood_Happy",
    "Mood_Neutral",
    "Mood_Stressed",
]

# ── Encoding maps (ordinal + one-hot) ───────────────────────────────
ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}
SLEEP_MAP = {"Poor": 0, "Moderate": 1, "Good": 2}
RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

# ── Human-readable feature labels (for reports) ─────────────────────
FEATURE_DISPLAY_NAMES = {
    "Heart_Rate": "Heart Rate (BPM)",
    "Blood_Pressure_Systolic": "Systolic Blood Pressure",
    "Blood_Pressure_Diastolic": "Diastolic Blood Pressure",
    "Stress_Level_Biosensor": "Biosensor Stress Level",
    "Stress_Level_Self_Report": "Self-Reported Stress Level",
    "Study_Hours": "Weekly Study Hours",
    "Project_Hours": "Weekly Project Hours",
    "Physical_Activity": "Physical Activity Level",
    "Sleep_Quality": "Sleep Quality",
    "Age": "Age",
    "Gender_F": "Gender (Female)",
    "Gender_M": "Gender (Male)",
    "Mood_Happy": "Mood (Happy)",
    "Mood_Neutral": "Mood (Neutral)",
    "Mood_Stressed": "Mood (Stressed)",
}
