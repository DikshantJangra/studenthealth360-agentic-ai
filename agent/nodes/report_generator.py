"""
Report Generator Node — Produces the final structured health report.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.prompts import REPORT_GENERATOR_SYSTEM_PROMPT, REPORT_GENERATOR_USER_PROMPT
from agent.factory import get_llm


def report_generator_node(state: AgentState) -> dict:
    """
    Generate the final structured health report using an LLM.
    """
    try:
        patient = state["patient_data"]
        guidelines = state.get("retrieved_guidelines", [])
        guidelines_text = "\n\n---\n\n".join(guidelines) if guidelines else "No specific guidelines retrieved."

        # Format guideline sources from metadata if available
        sources = state.get("guideline_sources", [])
        sources_text = "\n".join(f"[{i+1}] {src}" for i, src in enumerate(sources)) if sources else "Various university medical modules"

        user_prompt = REPORT_GENERATOR_USER_PROMPT.format(
            user_name=state.get("user_name", "Student"),
            primary_goal=state.get("primary_goal", "General Wellness"),
            risk_analysis=state.get("risk_analysis", "Patient risk analysis unavailable."),
            guidelines_text=guidelines_text,
            guideline_sources=sources_text,
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
        )

        llm = get_llm()
        messages = [
            SystemMessage(content=REPORT_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        return {"health_report": response.content.strip()}

    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}
