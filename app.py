"""
StudentHealth360 — Streamlit Application
AI-Powered Student Wellness Risk Assessment with Agentic AI Support.
"""

import streamlit as st
from ml.predict import predict
from agent.graph import run_agent

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudentHealth360",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background: #ffffff; }

    .main .block-container { padding: 2rem 3rem; max-width: 1200px; }

    /* Header */
    .header { text-align: center; margin-bottom: 2.5rem; }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #000 !important; margin: 0; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-top: 0.5rem; font-weight: 400; }
    .badge { display: inline-block; background: #f0fdf4; color: #16a34a; font-size: 0.75rem;
             font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 999px; margin-top: 0.5rem;
             border: 1px solid #bbf7d0; }

    /* Section titles */
    .section-title { font-size: 0.85rem; font-weight: 600; color: #374151; margin-bottom: 1.5rem;
                     text-transform: uppercase; letter-spacing: 0.05em; }

    /* Input styling */
    .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #374151 !important; font-weight: 400 !important; font-size: 0.9rem !important; }
    .stNumberInput input, .stSelectbox select {
        border: 1px solid #e5e7eb !important; border-radius: 6px !important; }

    /* Button */
    .stButton>button { background: #111827; color: white; font-weight: 500; border: none;
                       padding: 0.75rem 2rem; border-radius: 8px; font-size: 0.95rem;
                       transition: all 0.2s; width: 100%; }
    .stButton>button:hover { background: #1f2937; transform: translateY(-1px);
                              box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    /* Result cards */
    .result { border: 1px solid #e5e7eb; border-radius: 10px; padding: 2rem; margin-top: 2rem; }
    .result-title { font-size: 1.5rem; font-weight: 600; color: #111827; margin-bottom: 0.75rem; }

    .risk-low { border-left: 4px solid #10b981; background: #f0fdf4; }
    .risk-moderate { border-left: 4px solid #f59e0b; background: #fffbeb; }
    .risk-high { border-left: 4px solid #ef4444; background: #fef2f2; }

    /* Score display */
    .score-container { display: flex; align-items: center; gap: 1.5rem; margin: 1.5rem 0;
                       padding: 1rem 1.5rem; background: #f9fafb; border-radius: 8px; }
    .score-big { font-size: 2.5rem; font-weight: 700; }
    .score-label { font-size: 0.85rem; color: #6b7280; }
    .score-low { color: #10b981; }
    .score-moderate { color: #f59e0b; }
    .score-high { color: #ef4444; }

    /* Features */
    .feature-tag { display: inline-block; background: #f3f4f6; color: #374151; font-size: 0.8rem;
                   padding: 0.3rem 0.75rem; border-radius: 999px; margin: 0.25rem;
                   border: 1px solid #e5e7eb; }

    /* Report */
    .report-section { margin-top: 2rem; padding: 1.5rem; background: #fafafa;
                      border-radius: 8px; border: 1px solid #e5e7eb; }
    .report-section h2 { font-size: 1.1rem; font-weight: 600; color: #111827;
                         margin-bottom: 0.75rem; }

    /* Agent reasoning expander */
    .reasoning-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                     padding: 1.25rem; margin-top: 1rem; font-size: 0.9rem; color: #4b5563;
                     line-height: 1.6; }

    /* Footer */
    .footer { text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 3rem;
              padding-top: 1.5rem; border-top: 1px solid #e5e7eb; }

    hr { border: none; height: 1px; background: #e5e7eb; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1 class="main-title" style="color: #000000 !important;">StudentHealth360</h1>
    <p class="subtitle">AI-Powered Student Wellness Risk Assessment</p>
    <span class="badge"><i class="fas fa-robot"></i>&nbsp; Agentic AI · LangGraph · RAG</span>
</div>
""", unsafe_allow_html=True)

# ── Input Form ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title"><i class="fas fa-heartbeat"></i> Physiological</p>',
                unsafe_allow_html=True)
    age = st.number_input("Age", 17, 30, 20)
    heart_rate = st.number_input("Heart Rate (BPM)", 40.0, 150.0, 72.0)
    bp_systolic = st.number_input("Systolic BP", 80.0, 180.0, 120.0)
    bp_diastolic = st.number_input("Diastolic BP", 50.0, 120.0, 80.0,
                                    help="Lower blood pressure reading (normal: 60-80 mmHg)")
    stress_bio = st.slider("Biosensor Stress", 1.0, 10.0, 5.0,
                            help="Objective stress from wearable sensors (1=Low, 10=Extreme)")

with col2:
    st.markdown('<p class="section-title"><i class="fas fa-book"></i> Academic</p>',
                unsafe_allow_html=True)
    study_hours = st.number_input("Study Hours/Week", 0.0, 100.0, 30.0)
    project_hours = st.number_input("Project Hours/Week", 0.0, 100.0, 15.0)
    stress_self = st.slider("Self-Reported Stress", 1.0, 10.0, 5.0,
                             help="Subjective stress level (1=Relaxed, 10=Overwhelmed)")
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

with col3:
    st.markdown('<p class="section-title"><i class="fas fa-brain"></i> Psychological</p>',
                unsafe_allow_html=True)
    sleep = st.selectbox("Sleep Quality", ["Poor", "Moderate", "Good"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Stressed"])
    gender = st.selectbox("Gender", ["M", "F"])

st.markdown("<hr>", unsafe_allow_html=True)

# ── Analyse Button ──────────────────────────────────────────────────
if st.button("🔍  Analyse Risk Profile", use_container_width=True):
    # Assemble patient data
    patient_data = {
        "Age": age,
        "Heart_Rate": heart_rate,
        "Blood_Pressure_Systolic": bp_systolic,
        "Blood_Pressure_Diastolic": bp_diastolic,
        "Stress_Level_Biosensor": stress_bio,
        "Stress_Level_Self_Report": stress_self,
        "Physical_Activity": activity,
        "Sleep_Quality": sleep,
        "Mood": mood,
        "Gender": gender,
        "Study_Hours": study_hours,
        "Project_Hours": project_hours,
    }

    # ── Step 1: ML Prediction ───────────────────────────────────────
    with st.spinner("🧠 Running ML risk prediction..."):
        try:
            risk_score, risk_class, risk_label, top_features = predict(patient_data)
        except Exception as e:
            st.error(f"❌ ML prediction failed: {e}")
            st.stop()

    # ── Step 2: Agentic AI Pipeline ─────────────────────────────────
    with st.spinner("🤖 Agentic AI analysing your health profile — reasoning, retrieving guidelines, generating report..."):
        try:
            final_state = run_agent(
                patient_data=patient_data,
                risk_score=risk_score,
                risk_class=risk_class,
                risk_label=risk_label,
                top_features=top_features,
            )
        except Exception as e:
            st.error(f"❌ Agent pipeline failed: {e}")
            st.stop()

    # ── Step 3: Display Results ─────────────────────────────────────
    risk_css = f"risk-{risk_label.lower()}"
    score_css = f"score-{risk_label.lower()}"
    icon_map = {"Low": "fa-check-circle", "Moderate": "fa-exclamation-triangle", "High": "fa-exclamation-circle"}
    icon = icon_map.get(risk_label, "fa-info-circle")

    # Risk badge + score
    st.markdown(f"""
    <div class="result {risk_css}">
        <div class="result-title"><i class="fas {icon}"></i>&nbsp; {risk_label} Risk</div>
        <div class="score-container">
            <div>
                <div class="score-big {score_css}">{risk_score:.0%}</div>
                <div class="score-label">Confidence Score</div>
            </div>
            <div style="flex: 1;">
                <div class="score-label">Top Contributing Factors</div>
                <div style="margin-top: 0.5rem;">
                    {''.join(f'<span class="feature-tag">{f}</span>' for f in top_features)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Health report
    report = final_state.get("health_report", "")
    if report:
        st.markdown("---")
        st.markdown("### 📋 AI-Generated Health Report")
        st.markdown(report)

    # Agent reasoning (expandable)
    with st.expander("🔬 View Agent Reasoning Details", expanded=False):
        analysis = final_state.get("risk_analysis", "")
        if analysis:
            st.markdown("**Risk Analysis (LLM Reasoning):**")
            st.markdown(f'<div class="reasoning-box">{analysis}</div>', unsafe_allow_html=True)

        guidelines = final_state.get("retrieved_guidelines", [])
        if guidelines:
            st.markdown("**Retrieved Medical Guidelines (RAG):**")
            for i, g in enumerate(guidelines, 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(g[:500] + "..." if len(g) > 500 else g)

        sources = final_state.get("guideline_sources", [])
        if sources:
            st.markdown("**Guideline Sources:**")
            for s in sources:
                st.markdown(f"- {s}")

        if final_state.get("error"):
            st.warning(f"⚠️ Agent encountered an issue: {final_state['error']}")

    # Success animation for low risk
    if risk_label == "Low":
        st.balloons()

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p><i class="fas fa-shield-alt"></i>&nbsp; Educational tool only. Not a substitute for professional medical advice.</p>
    <p style="margin-top: 0.5rem; font-size: 0.75rem;">
        StudentHealth360 · Powered by LangGraph + Groq + ChromaDB · Milestone 2
    </p>
</div>
""", unsafe_allow_html=True)
