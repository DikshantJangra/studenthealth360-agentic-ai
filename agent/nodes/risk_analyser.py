"""
Risk Analyser Node — LLM-powered reasoning over patient risk data.
Produces a concise clinical analysis explaining the risk classification.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from agent.prompts import RISK_ANALYSER_SYSTEM_PROMPT, RISK_ANALYSER_USER_PROMPT
from config import GROQ_API_KEY, GROQ_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS


def risk_analyser_node(state: AgentState) -> dict:
    """
    Analyse the ML risk prediction and patient data using an LLM.

    Reads: patient_data, risk_score, risk_label, top_features
    Writes: risk_analysis (or error)
    """
    try:
        patient = state["patient_data"]

        # Format the user prompt with actual patient values
        user_prompt = RISK_ANALYSER_USER_PROMPT.format(
            risk_score=state["risk_score"],
            risk_label=state["risk_label"],
            top_features=", ".join(state["top_features"]),
            age=patient.get("Age", "N/A"),
            gender=patient.get("Gender", "N/A"),
            heart_rate=patient.get("Heart_Rate", "N/A"),
            bp_systolic=patient.get("Blood_Pressure_Systolic", "N/A"),
            bp_diastolic=patient.get("Blood_Pressure_Diastolic", "N/A"),
            stress_bio=patient.get("Stress_Level_Biosensor", "N/A"),
            stress_self=patient.get("Stress_Level_Self_Report", "N/A"),
            physical_activity=patient.get("Physical_Activity", "N/A"),
            sleep_quality=patient.get("Sleep_Quality", "N/A"),
            mood=patient.get("Mood", "N/A"),
            study_hours=patient.get("Study_Hours", "N/A"),
            project_hours=patient.get("Project_Hours", "N/A"),
        )

        # Call Groq LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        messages = [
            SystemMessage(content=RISK_ANALYSER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        analysis = response.content.strip()

        if not analysis:
            return {"error": "Risk analyser returned empty response."}

        return {"risk_analysis": analysis}

    except Exception as e:
        return {"error": f"Risk analyser failed: {str(e)}"}
