"""
Preprocessing utilities for patient input data.
"""

import pandas as pd
import joblib
from config import (
    SCALER_PATH,
    NUMERICAL_FEATURES,
    MODEL_FEATURE_ORDER,
    ACTIVITY_MAP,
    SLEEP_MAP,
)

# ── Singleton Scaler Loader ─────────────────────────────────────────
_scaler = None

def _get_scaler():
    """Load the fitted StandardScaler once and reuse it."""
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def preprocess_input(patient_data: dict) -> pd.DataFrame:
    """
    Convert raw patient input dict into a model-ready DataFrame.
    """
    # --- Ordinal encoding ---
    activity_encoded = ACTIVITY_MAP.get(patient_data["Physical_Activity"], 1)
    sleep_encoded = SLEEP_MAP.get(patient_data["Sleep_Quality"], 1)

    # --- One-hot encoding ---
    gender = patient_data.get("Gender", "M")
    mood = patient_data.get("Mood", "Neutral")

    row = {
        "Age": patient_data["Age"],
        "Heart_Rate": patient_data["Heart_Rate"],
        "Blood_Pressure_Systolic": patient_data["Blood_Pressure_Systolic"],
        "Blood_Pressure_Diastolic": patient_data["Blood_Pressure_Diastolic"],
        "Stress_Level_Biosensor": patient_data["Stress_Level_Biosensor"],
        "Stress_Level_Self_Report": patient_data["Stress_Level_Self_Report"],
        "Physical_Activity": activity_encoded,
        "Sleep_Quality": sleep_encoded,
        "Study_Hours": patient_data["Study_Hours"],
        "Project_Hours": patient_data["Project_Hours"],
        "Gender_F": 1 if gender == "F" else 0,
        "Gender_M": 1 if gender == "M" else 0,
        "Mood_Happy": 1 if mood == "Happy" else 0,
        "Mood_Neutral": 1 if mood == "Neutral" else 0,
        "Mood_Stressed": 1 if mood == "Stressed" else 0,
    }

    df = pd.DataFrame([row])

    # --- Standard scaling ---
    scaler = _get_scaler()
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    # --- Ensure column order ---
    return df[MODEL_FEATURE_ORDER]
