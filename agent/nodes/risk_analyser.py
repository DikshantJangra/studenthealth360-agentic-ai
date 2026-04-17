"""
Risk Analyser Node — LLM-powered reasoning over patient risk data.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.prompts import RISK_ANALYSER_SYSTEM_PROMPT, RISK_ANALYSER_USER_PROMPT
from agent.factory import get_llm


def risk_analyser_node(state: AgentState) -> dict:
    """
    Analyse the ML risk prediction and patient data using an LLM.
    """
    try:
        patient = state["patient_data"]

        user_prompt = RISK_ANALYSER_USER_PROMPT.format(
            user_name=state.get("user_name", "Student"),
            primary_goal=state.get("primary_goal", "Wellness"),
            journal_entry=state.get("journal_entry", "No entry"),
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

        llm = get_llm()
        messages = [
            SystemMessage(content=RISK_ANALYSER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        # Note: We store reasoning in a temporary key or just return it if generator needs it.
        # According to README, report_generator should use this. 
        # Let's add risk_analysis to AgentState if it's not there (it was in my previous version)
        return {"risk_analysis": response.content.strip()}

    except Exception as e:
        return {"error": f"Risk analyser failed: {str(e)}"}
