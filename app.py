"""
StudentHealth360 — Clinical Intelligence Interface
Compact, professional UI using native Streamlit components.
"""

import streamlit as st
import time
from ml.predict import predict
from agent.graph import run_agent

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="StudentHealth360 | Clinical Intelligence",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""<style>
    /* Tighten sidebar width */
    section[data-testid="stSidebar"] { width: 340px !important; }
    footer { display: none !important; }
    /* Remove gap between items in tabs */
    [data-testid="stVerticalBlock"] > div:has(div > [data-testid="stTabs"]) { gap: 0rem; }
</style>""")

# ── Sidebar Interface ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(":material/health_and_safety: **SH360** · Clinical Workspace")
    st.caption("Student health risk assessment powered by ML + RAG")

    # ── Student Profile ──
    with st.expander("Student profile", icon=":material/person:", expanded=True):
        u_name = st.text_input("Name", placeholder="Your name", help="Used to personalize your report.")
        u_goal = st.selectbox(
            "Wellness goal",
            ["Reduce Academic Stress", "Improve Sleep Quality", "Manage Anxiety", "Boost Physical Energy", "Balanced Lifestyle"],
            help="Your main priority right now.",
        )

    # ── Vitals ──
    with st.expander("Vital signs", icon=":material/monitor_heart:", expanded=True):
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            u_age = st.number_input("Age", 17, 45, 21)
            u_sys = st.number_input("Systolic BP", 80, 180, 120)
        with col_v2:
            u_gender = st.selectbox("Gender", ["M", "F", "Other"], index=1)
            u_dia = st.number_input("Diastolic BP", 50, 130, 80)
        u_hr = st.slider("Heart rate (BPM)", 40, 160, 72)
        u_bio_stress = st.slider("Biological stress index", 1, 10, 5, help="Biosensor-detected stress.")

    # ── Lifestyle ──
    with st.expander("Lifestyle & context", icon=":material/psychology:", expanded=False):
        u_sleep = st.select_slider("Sleep quality", ["Poor", "Fair", "Good", "Excellent"], value="Good")
        u_mood = st.selectbox("Mood", ["Happy", "Neutral", "Anxious", "Stressed", "Exhausted"])
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            u_study = st.number_input("Study hrs/wk", 0, 120, 35)
        with col_l2:
            u_project = st.number_input("Project hrs/wk", 0, 120, 15)
        u_activity = st.select_slider("Activity level", ["Low", "Moderate", "High"], value="Moderate")
        u_perceived_stress = st.slider("Perceived stress", 1, 10, 6, help="How stressed do you feel?")

    # ── Journal ──
    with st.expander("Weekly reflection", icon=":material/edit_note:", expanded=False):
        u_journal = st.text_area(
            "Journal entry",
            placeholder="How has your week been? Challenges or wins?",
            height=100,
        )

    st.write("")
    execute_diag = st.button("Run clinical engine", icon=":material/play_arrow:", use_container_width=True, type="primary")

# ── Main Workspace ──────────────────────────────────────────────────
if not execute_diag:
    # Landing state
    st.badge("System standby", icon=":material/radio_button_checked:", color="gray")
    st.title("Clinical interface")
    st.write("Complete the sidebar intake, then press **Run clinical engine** to generate your AI assessment.")

    st.write("")

    with st.container(border=True, horizontal_alignment="center"):
        st.write("")
        st.markdown(":material/biotech:", text_alignment="center")
        st.subheader("Engine offline", text_alignment="center")
        st.caption("Fill in your profile and vitals to begin.", text_alignment="center")
        st.write("")

else:
    # ── Orchestration ───────────────────────────────────────────────
    sleep_map = {"Poor": "Poor", "Fair": "Moderate", "Good": "Good", "Excellent": "Good"}
    patient_data = {
        "Age": u_age, "Heart_Rate": u_hr, "Blood_Pressure_Systolic": u_sys,
        "Blood_Pressure_Diastolic": u_dia, "Stress_Level_Biosensor": u_bio_stress,
        "Stress_Level_Self_Report": u_perceived_stress, "Physical_Activity": u_activity,
        "Sleep_Quality": sleep_map[u_sleep],
        "Mood": u_mood if u_mood in ["Happy", "Neutral", "Stressed"] else "Neutral",
        "Gender": u_gender[0] if u_gender in ["M", "F"] else "F",
        "Study_Hours": u_study, "Project_Hours": u_project,
    }

    with st.status("Initializing clinical protocols...", expanded=True) as status:
        st.write(":material/model_training: Running ML risk prediction...")
        risk_score, risk_class, risk_label, top_features = predict(patient_data)

        st.write(":material/smart_toy: Executing agentic reasoning workflow...")
        final_state = run_agent(
            user_name=u_name if u_name else "Student",
            primary_goal=u_goal,
            journal_entry=u_journal if u_journal else "No journal entry provided.",
            patient_data=patient_data,
            risk_score=risk_score,
            risk_label=risk_label,
            top_features=top_features,
        )
        status.update(label="Analysis complete", state="complete", expanded=False)

    # ── Handle Agent Errors ─────────────────────────────────────────
    if final_state.get("error"):
        st.error(f"Engine Error: {final_state['error']}", icon=":material/report_problem:")

    # ── Header row ──────────────────────────────────────────────────
    badge_color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(risk_label, "gray")
    badge_icon = {"Low": ":material/check_circle:", "Moderate": ":material/warning:", "High": ":material/error:"}.get(risk_label, ":material/info:")

    col_h1, col_h2 = st.columns([3, 1], vertical_alignment="bottom")
    with col_h1:
        st.badge(f"{risk_label} priority", icon=badge_icon, color=badge_color)
        st.title("Clinical findings")
    with col_h2:
        st.caption("Assessment UID")
        st.code(f"SH360-{int(time.time()):x}", language=None)

    # ── Metrics row ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subject", u_name if u_name else "Anonymous")
    m2.metric("Heart rate", f"{u_hr} BPM")
    m3.metric("Blood pressure", f"{u_sys}/{u_dia}")
    m4.metric("Confidence", f"{risk_score:.1%}")

    st.write("")

    # ── Primary analysis card ───────────────────────────────────────
    with st.container(border=True):
        st.subheader("Primary analysis")
        st.write(
            f"The diagnostic engine identifies **{risk_label}** health risk, "
            f"driven primarily by {', '.join([f'*{f}*' for f in top_features])}."
        )
        badge_md = " ".join([f":blue-badge[{f}]" for f in top_features])
        st.markdown(badge_md)

    # ── Detailed report tabs ────────────────────────────────────────
    tab_reasoning, tab_protocol, tab_evidence = st.tabs([
        ":material/psychology: Agentic reasoning",
        ":material/medical_services: Care protocol",
        ":material/library_books: Evidence registry",
    ])

    with tab_reasoning:
        reasoning = final_state.get("risk_analysis")
        if reasoning and str(reasoning).strip().lower() != "none":
            st.markdown(reasoning)
        else:
            st.info("Medical reasoning is being synthesized. This usually takes 5-10 seconds.", icon=":material/pending:")

    with tab_protocol:
        report = final_state.get("health_report")
        if report and str(report).strip().lower() != "none":
            st.markdown(report)
        else:
            st.info("Clinical protocol generation in progress...", icon=":material/pending:")

    with tab_evidence:
        guidelines = final_state.get("retrieved_guidelines", [])
        if guidelines:
            for i, g in enumerate(guidelines, 1):
                with st.container(border=True):
                    st.caption(f"Evidence point #{i}")
                    st.write(g)
        else:
            st.info("No external medical guidelines were prioritized for this profile.", icon=":material/info:")

    if risk_label == "Low":
        st.balloons()

    # ── Disclaimer ──────────────────────────────────────────────────
    st.write("")
    st.divider()
    st.caption(
        "**Disclaimer:** This clinical assessment is generated by an experimental AI system. "
        "It is designed for screening and support purposes only and does not replace "
        "professional medical diagnosis or consultation."
    )

# ── Footer ──────────────────────────────────────────────────────────
st.write("")
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("**STUDENTHEALTH360**")
with col_f2:
    st.caption("Precision wellness for the academic journey", text_alignment="center")
with col_f3:
    st.caption("© 2026 NST · Clinical Protocol v3.0", text_alignment="right")
