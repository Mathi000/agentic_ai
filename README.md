# 🧠 Workforce Intelligence Dashboard (Agentic AI)

A state-of-the-art AI-powered workforce management platform that automates attendance reporting, predicts employee risk, and provides generative AI insights.

---

## 🔥 Key Features

*   **📊 Dynamic Holographic Dashboard**: Visualizes workforce strength, risk profiles, and behavioral clusters.
*   **🧠 Predictive Risk Engine**: Uses XGBoost and Scikit-Learn to proactively identify "at-risk" employees based on attendance patterns.
*   **🤖 HR Intelligence Assistant**: A Natural Language (LLM) powered chat for querying workforce data in plain English.
*   **👥 Integrated Employee Portal**: Self-service leave management with automated balance tracking.
*   **⚡ One-Click Automation**: Converts messy biometric/master attendance CSVs into standardized reports.

---

## 🚀 Quick Start

### 📋 Prerequisites
1.  **Python 3.9+** installed.
2.  **Groq Cloud API Key** (for the AI Chat feature).

### 🛠️ Setup
1.  **Clone/Extract** the project folder.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Key**: Create a `.env` file in the root directory and add:
    ```env
    GROQ_API_KEY=your_api_key_here
    ```

### 🏃 Running the App
The easiest way to start is by using the provided launcher:
1.  Double-click on **`Launch_HR_Dashboard.bat`**.
2.  Log in as:
    *   **HR**: ID: `hr` / Passphrase: `hr`
    *   **Employee**: ID: `emp` / Passphrase: `emp`

---

## 📂 Project Structure

*   `dashboard/app.py`: Main Streamlit interface.
*   `core/`: Core AI engines (Risk, Pattern, Agent, LLM).
*   `data/`: Data storage for leave applications and balances.
*   `reports/`: Automated daily summary CSV outputs.

---

## 🛠️ Tech Stack
*   **Frontend**: Streamlit
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn, XGBoost, Joblib
*   **Generative AI**: Groq (Llama-3), Python-Dotenv
*   **Visuals**: Plotly Express, Custom CSS (Aesthetic 2.0)

---

## 📋 Note on Data Ingestion
Simply drop the **Master Attendance CSV** into the `data/` folder or upload it through the HR dashboard. The system will automatically handle parsing, cleanup, and feature engineering.