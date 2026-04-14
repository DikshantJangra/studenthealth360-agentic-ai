"""
Report Generator Node — Produces the final structured health report.
Combines risk analysis + retrieved guidelines into a 4-section report.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from agent.prompts import (
    REPORT_GENERATOR_SYSTEM_PROMPT,
    REPORT_GENERATOR_USER_PROMPT,
)
from config import GROQ_API_KEY, GROQ_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS


def report_generator_node(state: AgentState) -> dict:
    """
    Generate the final structured health report using an LLM.

    Reads: risk_analysis, retrieved_guidelines, guideline_sources,
           risk_score, risk_label, top_features, patient_data
    Writes: health_report (or error)
    """
    try:
        patient = state["patient_data"]

        # Combine retrieved guidelines into a single text block
        guidelines = state.get("retrieved_guidelines", [])
        guidelines_text = "\n\n---\n\n".join(guidelines) if guidelines else \
            "No specific guidelines retrieved. Use general wellness best practices."

        # Format guideline sources
        sources = state.get("guideline_sources", [])
        sources_text = "\n".join(
            f"[{i+1}] {src}" for i, src in enumerate(sources)
        ) if sources else "No specific sources available."

        # Use risk analysis or provide fallback
        risk_analysis = state.get("risk_analysis", "")
        if not risk_analysis:
            risk_analysis = (
                f"Patient classified as {state['risk_label']} risk with a "
                f"score of {state['risk_score']:.2f}. Top factors: "
                f"{', '.join(state['top_features'])}."
            )

        # Format the user prompt
        user_prompt = REPORT_GENERATOR_USER_PROMPT.format(
            risk_analysis=risk_analysis,
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
            SystemMessage(content=REPORT_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        report = response.content.strip()

        if not report:
            return {"error": "Report generator returned empty response."}

        return {"health_report": report}

    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}
