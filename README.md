# MILESTONE 2: AGENTIC AI HEALTH SUPPORT ASSISTANT (END-SEM SUBMISSION)
## Intelligent Patient Risk Assessment & Agentic Health Support System - Student Health Assessment 360

---

## Objective
Extend the risk assessment system into an **agentic AI health support assistant** that autonomously reasons over patient data, retrieves medical knowledge, and generates structured health guidance.

---

## 🛠 Functional Requirements
- **Autonomous Analysis:** Autonomously analyze risk predictions and patient profiles.
- **Knowledge Retrieval:** Retrieve medical guidelines and contextually relevant content via RAG.
- **Structured Reporting:** Generate comprehensive and structured health summaries.
- **Graceful Handling:** Handle incomplete data or tool failures with robust fallback mechanisms.

## 🧠 Technical Requirements (Agentic)
- **Framework:** LangGraph (for complex workflow and shared state management).
- **RAG System:** Vector DB (Chroma/FAISS) for indexing and retrieving medical guidelines.
- **State Management:** Explicit state management across multiple reasoning steps.
- **Hallucination Guard:** Implementation of strategies to reduce hallucinations and ensure grounding.

---

## 📋 Structured Output
The agent generates a report containing:
- **Risk Summary:** Detailed patient risk profile and identified key factors.
- **Recommendations:** Actionable preventive care and follow-up suggestions.
- **Sources:** Proper attribution to medical guidelines and retrieved chunks.
- **Disclaimer:** Mandatory medical advice disclaimer (Non-diagnostic).

---

---

## Live Demo

🔗 **Deployed App:** [Link to Hugging Face Spaces / Streamlit Cloud]
📁 **GitHub Repo:** [github.com/DikshantJangra/StudentHealthAssessment360](https://github.com/DikshantJangra/StudentHealthAssessment360)
🎥 **Demo Video:** [Google Drive](https://drive.google.com/drive/folders/1I9yAZUiRHLviQuqN2Y2R37N94Y-8p8Ao?usp=sharing)

---

## System Architecture

```
Patient Input (Streamlit UI)
        │
        ▼
ML Risk Predictor ← Milestone 1 model (logistic regression)
        │  risk score + top features
        ▼
LangGraph Agent Entry Node
  ├── Risk Analyser Node       → LLM reasons over risk score + patient features
  ├── RAG Retriever Node       → Chroma/FAISS fetches relevant medical guidelines
  └── Hallucination Guard      → Prompt grounding + source attribution
        │
        ▼
Report Generator Node
  └── Structured health report:
        ├── Risk Summary
        ├── Preventive Recommendations
        ├── Cited Guidelines (sources)
        └── Medical Disclaimer
        │
        ▼
Error / Fallback Handler (conditional LangGraph edge)
        │
        ▼
Streamlit Output UI
        │
        ▼
Hosted on Hugging Face Spaces
```

---

## Agent State

The LangGraph graph passes a typed `AgentState` through every node:

```python
from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    patient_data: dict
    risk_score: float
    top_features: List[str]
    retrieved_guidelines: List[str]
    health_report: Optional[str]
    error: Optional[str]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq API (Llama 3, free tier) |
| Vector DB / RAG | ChromaDB + `all-MiniLM-L6-v2` embeddings |
| ML Model | Scikit-learn (from Milestone 1) |
| UI | Streamlit |
| Hosting | Hugging Face Spaces |
| Data | Pandas, NumPy |
| Env Management | `python-dotenv` |

---

## Project Structure

```
StudentHealthAssessment360/
│
├── app.py                        # Streamlit entry point
│
├── agent/
│   ├── graph.py                  # LangGraph graph definition (nodes + edges)
│   ├── state.py                  # AgentState TypedDict
│   ├── nodes/
│   │   ├── risk_analyser.py      # LLM reasoning over risk score
│   │   ├── rag_retriever.py      # Chroma retrieval of guidelines
│   │   ├── report_generator.py   # Structured report output
│   │   └── error_handler.py      # Fallback / graceful degradation
│   └── prompts.py                # All system + user prompt templates
│
├── rag/
│   ├── build_vectorstore.py      # Ingest + embed medical guidelines
│   ├── guidelines/               # Raw .txt guideline documents
│   └── vectorstore/              # Persisted Chroma DB
│
├── ml/
│   ├── model.pkl                 # Trained M1 model (serialised)
│   ├── predict.py                # predict(patient_data) → risk_score, features
│   └── preprocessing.py          # Scaler, encoder (reused from M1)
│
├── data/
│   └── student_health_dataset.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb       # From Milestone 1
│   └── data_modelling.ipynb      # From Milestone 1
│
├── requirements.txt
├── .env.example                  # API key template (never commit .env)
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/DikshantJangra/StudentHealthAssessment360.git
cd StudentHealthAssessment360
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Add your Groq API key inside .env:
# GROQ_API_KEY=your_key_here
```

### 4. Build the RAG vector store (first time only)

```bash
python rag/build_vectorstore.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
langgraph
langchain-groq
chromadb
sentence-transformers
scikit-learn
pandas
numpy
python-dotenv
```

---

## How the Agent Works

**Step 1 — Input**
User fills in patient details (age, vitals, lab values, medical history) in the Streamlit UI.

**Step 2 — ML Prediction**
The Milestone 1 model runs inference and returns a risk score (0–1) and the top 3 contributing features.

**Step 3 — LangGraph takes over**
The risk score and features are loaded into `AgentState` and the graph begins execution across three parallel node branches.

**Step 4 — RAG Retrieval**
The `rag_retriever` node queries ChromaDB using the predicted risk category (e.g. "high cardiovascular risk") and pulls the 3 most semantically relevant guideline chunks from the embedded medical knowledge base.

**Step 5 — LLM Reasoning**
The `risk_analyser` and `report_generator` nodes call the Groq LLM. Prompts explicitly include the retrieved guidelines to prevent hallucination. The model is instructed to only reference provided sources and to flag uncertainty.

**Step 6 — Structured Report**
The final report contains four mandatory sections: Risk Summary, Preventive Recommendations, Cited Sources, and a Medical Disclaimer.

**Step 7 — Error Handling**
A conditional LangGraph edge routes to the `error_handler` node if the LLM call fails or retrieved guidelines are empty — ensuring graceful degradation rather than a crash.

---

## Sample Output

```
RISK SUMMARY
Patient presents a HIGH risk score of 0.82, driven primarily by elevated
BMI, irregular sleep patterns, and high reported stress levels.

PREVENTIVE RECOMMENDATIONS
1. Consult a physician for a formal cardiovascular screening.
2. Initiate a structured sleep hygiene programme (7–9 hrs/night).
3. Reduce processed food intake; follow a balanced diet per WHO guidelines.

SOURCES
[1] WHO — Global Action Plan for the Prevention of NCDs (2013–2020)
[2] CDC — Sleep and Sleep Disorders: Recommendations for Adults
[3] AHA — Cardiovascular Risk Reduction Guidelines
```

---

## Ethical AI Considerations

- **No hallucination:** LLM outputs are grounded exclusively in retrieved guideline chunks via RAG. Prompts explicitly prohibit the model from generating unsourced medical claims.
- **Mandatory disclaimer:** Every report includes a medical advice disclaimer — in both the generated text and the Streamlit UI.
- **Data privacy:** No patient data is stored or logged. All inference is session-local.
- **Bias awareness:** The M1 model was trained with feature selection to remove non-predictive demographic proxies.

---

## 🏆 Evaluation Criteria

| Phase | Weight | Criteria |
|---|---|---|
| **Mid-Sem (Milestone 1)** | 25% | Correct application of ML concepts, Quality of data preprocessing/Feature selection, Model performance, Code modularity & UI usability. |
| **End-Sem (Milestone 2)** | 30% | Quality/reliability of agent reasoning, Correct RAG implementation & State management, Clarity/Structure of health reports, Ethical AI & Deployment success. |

---

## 📦 End-Sem Deliverables
- **Hosted Link:** Fully deployed public application.
- **GitHub Repository:** Complete, well-documented codebase.
- **Demo Video:** Thorough system walkthrough and demo.
- **Design Docs:** Agent workflow and architecture design documents.

---

## 👥 Team

| Name | Enrollment Number | Email ID | Role |
|---|---|---|---|
| **Dikshant Jangra** (Lead) | 2401010155 | dikshant.jangra2024@nst.rishihood.edu.in | ML pipeline, LangGraph architecture, deployment |
| **Mukul Kumar** | 2401010285 | mukul.kumar2024@nst.rishihood.edu.in | RAG setup, ChromaDB, embeddings |
| **Vaibav Vats** | 2401020112 | vaibhav.v2024@nst.rishihood.edu.in | Streamlit UI, report formatting |
| **Ashish Kumar Yadav** | 2401020015 | ashish.yadav2024@nst.rishihood.edu.in | Prompt engineering, evaluation, viva prep |

---

## Milestone 1 Reference

> The data cleaning and modelling notebooks, original dataset, logistic regression model, and initial Streamlit UI were completed in **Milestone 1**. See [`/notebooks`](./notebooks/) and the [M1 Colab links](https://colab.research.google.com/drive/1mzHsZ85XsmSOWcDOzpGTqQdcqMXKR1yC) for full details. Milestone 2 builds directly on top of that work without replacing it.

---

*Project 3 — AI/ML · Intelligent Patient Risk Assessment & Agentic Health Support System*
