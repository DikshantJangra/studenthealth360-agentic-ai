"""
ML prediction interface.
Wraps the Milestone 1 logistic-regression model for risk inference.
"""

import joblib
import numpy as np
from typing import Tuple, List

from config import MODEL_PATH, MODEL_FEATURE_ORDER, RISK_LABELS, FEATURE_DISPLAY_NAMES
from ml.preprocessing import preprocess_input

# ── Singleton Model Loader ──────────────────────────────────────────
_model = None

def _get_model():
    """Load the trained model once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(patient_data: dict) -> Tuple[float, int, str, List[str]]:
    """
    Run risk prediction on raw patient data.

    Parameters
    ----------
    patient_data : dict
        Raw patient input (Age, Heart_Rate, …, Gender, Mood).

    Returns
    -------
    risk_score : float
        Probability of the predicted risk class (0.0 – 1.0).
    risk_class : int
        Predicted class index (0 = Low, 1 = Moderate, 2 = High).
    risk_label : str
        Human-readable risk label.
    top_features : list[str]
        Top 3 features most responsible for this prediction.
    """
    model = _get_model()
    df = preprocess_input(patient_data)

    # --- Prediction ----------------------------------------------------------
    risk_class = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    risk_score = float(probabilities[risk_class])
    risk_label = RISK_LABELS[risk_class]

    # --- Top contributing features -------------------------------------------
    top_features = _extract_top_features(model, df, risk_class, k=3)

    return risk_score, risk_class, risk_label, top_features


def _extract_top_features(model, df, risk_class: int, k: int = 3) -> List[str]:
    """
    Extract top-k features driving the prediction using model coefficients.
    """
    try:
        # Logistic regression: coef_ shape is (n_classes, n_features)
        coefficients = model.coef_[risk_class]
        feature_values = df.values[0]

        # Contribution ≈ |coefficient × feature_value|
        contributions = np.abs(coefficients * feature_values)
        top_indices = np.argsort(contributions)[::-1][:k]

        feature_names = MODEL_FEATURE_ORDER
        top = []
        for idx in top_indices:
            raw_name = feature_names[idx]
            display = FEATURE_DISPLAY_NAMES.get(raw_name, raw_name)
            top.append(display)
        return top

    except (AttributeError, IndexError):
        # Fallback: return features with highest absolute values
        feature_values = np.abs(df.values[0])
        top_indices = np.argsort(feature_values)[::-1][:k]
        return [
            FEATURE_DISPLAY_NAMES.get(MODEL_FEATURE_ORDER[i], MODEL_FEATURE_ORDER[i])
            for i in top_indices
        ]
