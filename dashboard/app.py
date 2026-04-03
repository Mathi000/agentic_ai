import sys
import os
import uuid
import json

# Ensure project root is on the path so core.* imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.data_engine import build_feature_pipeline
from core.pattern_engine import run_pattern_engine
from core.risk_engine import predict_risk
from core.agent_engine import run_agent_layer
from core.llm_engine import classify_query, filter_context, generate_response, _load_outputs


# =========================================================
# 🎨 Page Config & Custom CSS
# =========================================================
st.set_page_config(
    page_title="Workforce Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    h1, h2, h3, h4, .kpi-value, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* ---- Jaw-Dropping Aurora/Cyber-AI Background Effect ---- */
    .stApp {
        background-color: #03050C;
        color: #f8fafc;
        overflow-x: hidden;
    }

    /* ---- DYNAMIC HOLOGRAPHIC ORBS ---- */
    .stApp::before {
        content: "";
        position: fixed;
        width: 150vw; height: 150vh;
        top: -25vh; left: -25vw;
        background: 
            radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.5), transparent 30%),
            radial-gradient(circle at 85% 75%, rgba(0, 245, 255, 0.5), transparent 30%),
            radial-gradient(circle at 50% 50%, rgba(255, 0, 127, 0.25), transparent 40%),
            radial-gradient(circle at 75% 25%, rgba(57, 255, 20, 0.2), transparent 40%);
        filter: blur(120px);
        animation: cyberOrbs 25s infinite alternate cubic-bezier(0.4, 0, 0.2, 1);
        z-index: -2;
        pointer-events: none;
        mix-blend-mode: screen;
    }

    /* HIGH-TECH ENERGY GRID */
    .stApp::after {
        content: "";
        position: fixed;
        width: 100vw; height: 100vh;
        top: 0; left: 0;
        background-image: 
            linear-gradient(rgba(0, 245, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 245, 255, 0.04) 1px, transparent 1px);
        background-size: 60px 60px;
        z-index: -1;
        pointer-events: none;
        animation: gridMove 20s linear infinite;
        opacity: 0.8;
    }

    @keyframes cyberOrbs {
        0% { transform: scale(1) translate(0, 0) rotate(0deg); opacity: 0.7; }
        50% { transform: scale(1.15) translate(3vw, 3vh) rotate(5deg); opacity: 1; }
        100% { transform: scale(0.85) translate(-3vw, -5vh) rotate(-5deg); opacity: 0.7; }
    }
    
    @keyframes gridMove {
        from { background-position: 0 0; }
        to { background-position: 0 600px; }
    }

    /* ---- Element Entrance Animations ---- */
    @keyframes fluidUp {
        0% { opacity: 0; transform: translateY(40px) scale(0.95); filter: blur(5px); }
        100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }
    @keyframes glowText {
        0%, 100% { text-shadow: 0 0 10px rgba(0,245,255,0.4), 0 0 20px rgba(0,245,255,0.2); }
        50% { text-shadow: 0 0 20px rgba(0,245,255,0.8), 0 0 40px rgba(0,245,255,0.4); }
    }

    .block-container { 
        padding-top: 1.5rem !important; 
        max-width: 1400px; 
        z-index: 2;
        animation: fluidUp 1s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* ---- Holographic KPI Cards ---- */
    .kpi-wrapper {
        background: rgba(10, 15, 30, 0.6);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 3rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(0, 245, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 245, 255, 0.05);
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: fluidUp 0.8s backwards;
    }

    div[data-testid="column"]:nth-child(1) .kpi-wrapper { animation-delay: 0.1s; }
    div[data-testid="column"]:nth-child(2) .kpi-wrapper { animation-delay: 0.2s; }
    div[data-testid="column"]:nth-child(3) .kpi-wrapper { animation-delay: 0.3s; }
    div[data-testid="column"]:nth-child(4) .kpi-wrapper { animation-delay: 0.4s; }

    /* Moving Laser Border Effect */
    .kpi-wrapper::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; right: 0; bottom: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00F5FF, transparent);
        animation: laserScan 4s linear infinite;
        opacity: 0;
        transition: opacity 0.4s;
    }
    @keyframes laserScan {
        from { left: -100%; }
        to { left: 100%; }
    }

    .kpi-wrapper:hover {
        transform: translateY(-10px) scale(1.05) rotateX(5deg);
        box-shadow: 0 30px 60px rgba(0, 245, 255, 0.25), inset 0 0 40px rgba(0, 245, 255, 0.15);
        border-color: rgba(0, 245, 255, 0.7);
        perspective: 1000px;
    }
    .kpi-wrapper:hover::before { opacity: 1; }

    .kpi-value { 
        color: #ffffff;
        animation: glowText 4s infinite alternate;
        margin: 0; 
        font-size: clamp(3rem, 5vw, 5.5rem); 
        font-weight: 900;
        line-height: 1;
        letter-spacing: -1px;
    }
    
    .kpi-label { 
        color: #94a3b8; 
        margin-top: 1.5rem; 
        font-size: 0.95rem; 
        text-transform: uppercase; 
        letter-spacing: 5px; 
        font-weight: 600;
    }

    /* ---- Massive Cinematic Hero Section ---- */
    .hero-container {
        text-align: center; 
        padding: 7rem 2rem; 
        background: radial-gradient(circle at center, rgba(10, 15, 30, 0.8) 0%, rgba(3, 10, 23, 0.9) 100%);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(0, 245, 255, 0.15);
        border-radius: 40px; 
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 40px 80px rgba(0, 0, 0, 0.8), inset 0 0 50px rgba(0, 245, 255, 0.05);
        animation: fluidUp 1s ease-out backwards;
    }

    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4.5rem); 
        font-weight: 900; 
        letter-spacing: -1px; 
        background: linear-gradient(180deg, #FFFFFF 0%, #00F5FF 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 2rem;
        line-height: 1.1;
        text-transform: uppercase;
        animation: glowText 5s infinite alternate;
    }
    
    .hero-subtitle {
        font-size: clamp(1.2rem, 3vw, 1.5rem); 
        color: #e2e8f0; 
        max-width: 900px; 
        margin: 0 auto 3rem auto; 
        line-height: 1.8; 
        font-weight: 300;
        letter-spacing: 0.5px;
        text-align: center !important;
        display: block;
    }

    /* ---- Section Headers ---- */
    .header-style {
        font-size: 2.2rem;
        font-weight: 900;
        margin-top: 4rem;
        margin-bottom: 2.5rem;
        color: #ffffff;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        border-bottom: 2px solid rgba(0, 245, 255, 0.2);
        padding-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0,245,255,0.4);
        animation: fluidUp 0.8s backwards;
    }

    /* ---- Glowing Data Boxes ---- */
    .insight-box {
        background: rgba(10, 15, 30, 0.8);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        line-height: 1.7;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.4s ease-out;
        color: #e2e8f0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.02);
        animation: fluidUp 0.6s backwards;
        animation-delay: 0.4s;
    }
    
    .insight-alert {
        border-left: 4px solid #ff007f;
        background: linear-gradient(90deg, rgba(255, 0, 127, 0.05) 0%, transparent 100%);
    }
    
    .insight-rec {
        border-left: 4px solid #00F5FF;
        background: linear-gradient(90deg, rgba(0, 245, 255, 0.05) 0%, transparent 100%);
    }

    .insight-box:hover {
        transform: scale(1.03) translateX(10px);
        border-color: rgba(0, 245, 255, 0.3);
        box-shadow: -10px 15px 40px rgba(0, 245, 255, 0.1);
    }

    /* ---- Stunning Neon Buttons & Inputs ---- */
    button[kind="primary"] {
        background: transparent !important;
        color: #00F5FF !important;
        border: 2px solid #00F5FF !important;
        border-radius: 30px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.2), inset 0 0 10px rgba(0, 245, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    }
    
    button[kind="primary"]:hover {
        background: #00F5FF !important;
        color: #030a17 !important;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.6), inset 0 0 20px rgba(0, 245, 255, 0.4) !important;
        transform: scale(1.05) translateY(-2px) !important;
    }

    /* Regular buttons */
    button[kind="secondary"] {
        background: rgba(10, 15, 30, 0.8) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 30px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        border-color: #8a2be2 !important;
        color: #8a2be2 !important;
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.4) !important;
        transform: scale(1.02) !important;
    }

    /* Inputs */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: rgba(10, 15, 30, 0.8) !important;
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: #00F5FF !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.5) !important;
        background-color: rgba(5, 10, 20, 0.9) !important;
    }

    /* ---- Holographic Sidebar & Chat ---- */
    section[data-testid="stSidebar"] {
        background-color: rgba(3, 10, 23, 0.9) !important;
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(0, 245, 255, 0.15);
        box-shadow: 5px 0 30px rgba(0,0,0,0.8);
    }
    
    .stChatMessage {
        background: rgba(10, 15, 30, 0.8);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 245, 255, 0.1);
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }

    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border-radius: 15px !important;
        overflow: hidden !important;
        border: 1px solid rgba(0, 245, 255, 0.2) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 15px rgba(0, 245, 255, 0.1) !important;
        animation: fluidUp 0.8s backwards;
        animation-delay: 0.6s;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# 🗂️ Session State Defaults
# =========================================================
for key, default in {
    "pipeline_done": False,
    "risk_df": None,
    "pattern_df": None,
    "feature_df": None,
    "agent_output": None,
    "chat_history": [{"role": "assistant", "content": "👋 Hi! I'm your Workforce AI Assistant. I can help analyze attendance patterns, flag anomalies, and summarize employee risks. How can I help you today?"}],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def get_data_paths():
    leaves_path = os.path.join(os.path.dirname(__file__), "..", "data", "leave_applications.csv")
    bals_path = os.path.join(os.path.dirname(__file__), "..", "data", "leave_balances.json")
    return leaves_path, bals_path

def save_leave_data():
    leaves_path, bals_path = get_data_paths()
    st.session_state.leave_applications.to_csv(leaves_path, index=False)
    with open(bals_path, "w") as f:
        json.dump(st.session_state.leave_balances, f)

if "leave_applications" not in st.session_state:
    leaves_path, _ = get_data_paths()
    if os.path.exists(leaves_path):
        st.session_state.leave_applications = pd.read_csv(leaves_path, dtype={"Employee ID": str})
    else:
        st.session_state.leave_applications = pd.DataFrame(columns=["Leave ID", "Employee ID", "Emp Name", "Start Date", "End Date", "Reason", "Status"])

if "leave_balances" not in st.session_state:
    _, bals_path = get_data_paths()
    if os.path.exists(bals_path):
        with open(bals_path, "r") as f:
            st.session_state.leave_balances = json.load(f)
    else:
        st.session_state.leave_balances = {}


# =========================================================
# ⚙️ Pipeline Orchestration
# =========================================================
def load_and_run_pipeline(file_path: str):
    with st.spinner("🚀 Initializing AI Engine & Processing Data..."):
        feature_df = build_feature_pipeline(file_path)
        pattern_df = run_pattern_engine(feature_df)
        risk_df = predict_risk(feature_df)
        agent_output = run_agent_layer(risk_df)
    return feature_df, pattern_df, risk_df, agent_output

# Base Plotly template for dark mode
PLOT_BG = "rgba(0,0,0,0)"
CHART_HEIGHT = 350

def get_base_layout():
    return dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Outfit", color="#94a3b8", size=13),
        margin=dict(t=30, b=30, l=10, r=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        height=CHART_HEIGHT
    )

# =========================================================
# 📊 Dashboard Rendering
# =========================================================
def render_dashboard(risk_df: pd.DataFrame, pattern_df: pd.DataFrame, agent_output: dict):
    # ---- KPI Row ----
    total = f"{len(risk_df):,}"
    high_count = f"{len(risk_df[risk_df['risk_category'] == 'High']):,}"
    anomaly_val = len(pattern_df[pattern_df["anomaly_flag"] == "Anomaly"]) if "anomaly_flag" in pattern_df.columns else 0
    anomaly_count = f"{anomaly_val:,}"
    avg_risk_val = risk_df["risk_score"].mean() if "risk_score" in risk_df.columns else 0
    avg_risk = f"{avg_risk_val:.2f}"

    k1, k2, k3, k4 = st.columns(4)
    for col, value, label in [
        (k1, total, "Total Workforce"),
        (k2, high_count, "High Risk"),
        (k3, anomaly_count, "Anomalies"),
        (k4, avg_risk, "Avg Risk Score"),
    ]:
        col.markdown(
            f'<div class="kpi-wrapper"><p class="kpi-value">{value}</p><p class="kpi-label">{label}</p></div>',
            unsafe_allow_html=True,
        )

    # ---- Charts Row 1 ----
    st.markdown('<div class="header-style">📊 Risk & Department Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.5])

    with c1:
        risk_counts = risk_df["risk_category"].value_counts().reset_index()
        risk_counts.columns = ["Risk Category", "Count"]
        color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        
        fig = px.pie(
            risk_counts, names="Risk Category", values="Count",
            color="Risk Category", color_discrete_map=color_map, hole=0.55
        )
        fig.update_layout(**get_base_layout())
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "department" in risk_df.columns:
            dept_risk = risk_df.groupby(["department", "risk_category"]).size().reset_index(name="count")
            fig2 = px.bar(
                dept_risk, x="department", y="count", color="risk_category",
                color_discrete_map=color_map, barmode="stack", text_auto=True
            )
            fig2.update_layout(**get_base_layout(), xaxis_title="", yaxis_title="Employees", legend_title="")
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Charts Row 2 ----
    st.markdown('<div class="header-style">🧠 Behavioral & Trend Intelligence</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)

    with c3:
        if "cluster_label" in pattern_df.columns:
            cluster_counts = pattern_df["cluster_label"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig3 = px.bar(
                cluster_counts, x="Cluster", y="Count", color="Cluster",
                color_discrete_sequence=["#8b5cf6", "#ec4899", "#3b82f6"]
            )
            fig3.update_layout(**get_base_layout(), showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        if "anomaly_flag" in pattern_df.columns:
            anom_counts = pattern_df["anomaly_flag"].value_counts().reset_index()
            anom_counts.columns = ["Status", "Count"]
            fig4 = px.pie(
                anom_counts, names="Status", values="Count", color="Status",
                color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"}, hole=0.55
            )
            fig4.update_layout(**get_base_layout())
            fig4.update_traces(marker=dict(line=dict(color='#0f172a', width=2)))
            st.plotly_chart(fig4, use_container_width=True)

    with c5:
        if "trend_flag" in pattern_df.columns:
            trend_counts = pattern_df["trend_flag"].value_counts().reset_index()
            trend_counts.columns = ["Trend", "Count"]
            fig5 = px.bar(
                trend_counts, y="Trend", x="Count", color="Trend", orientation='h',
                color_discrete_sequence=["#f97316", "#ef4444", "#06b6d4", "#14b8a6"]
            )
            fig5.update_layout(**get_base_layout(), showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig5, use_container_width=True)

    # ---- Employee Risk Table ----
    st.markdown('<div class="header-style">📋 Deep Data Inspector</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.multiselect("Risk Level Filter", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with f2:
        dept_filter = None
        if "department" in risk_df.columns:
            dept_filter = st.multiselect("Department Filter", sorted(risk_df["department"].dropna().unique()), default=sorted(risk_df["department"].dropna().unique()))

    filtered = risk_df[risk_df["risk_category"].isin(risk_filter)]
    if dept_filter and "department" in filtered.columns:
        filtered = filtered[filtered["department"].isin(dept_filter)]
    if "risk_score" in filtered.columns:
        filtered = filtered.sort_values("risk_score", ascending=False)

    st.dataframe(
        filtered,
        use_container_width=True,
        height=400,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.3f", min_value=0, max_value=1),
            "risk_category": st.column_config.TextColumn("Category"),
            "employee_id": st.column_config.TextColumn("ID"),
            "Emp Name": st.column_config.TextColumn("Employee Name"),
        },
    )

    # ---- Agent Insights ----
    st.markdown('<div class="header-style">⚡ Autonomous Action Plan</div>', unsafe_allow_html=True)

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("<h4 style='color:#f8fafc; margin-bottom: 1rem;'>🚨 High Priority Alerts</h4>", unsafe_allow_html=True)
        for alert in agent_output.get("alerts", []):
            st.markdown(f'<div class="insight-box insight-alert">{alert}</div>', unsafe_allow_html=True)

    with i2:
        st.markdown("<h4 style='color:#f8fafc; margin-bottom: 1rem;'>💡 Strategic Recommendations</h4>", unsafe_allow_html=True)
        for rec in agent_output.get("recommendations", []):
            st.markdown(f'<div class="insight-box insight-rec">{rec}</div>', unsafe_allow_html=True)


# =========================================================
# 💬 Chat Interface (Using Native st.chat_message)
# =========================================================
def chat_interface():
    st.markdown('<div class="header-style">💬 HR Intelligence Assistant</div>', unsafe_allow_html=True)
    st.caption("Interact with your data using natural language. Ensure you have configured your `.env` API key.")
    
    # Render interactive chat history
    for msg in st.session_state.chat_history:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat Input Box
    if prompt := st.chat_input("Ask about high risk employees, specific departments, or behavioral trends..."):
        # Add User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # Process Bot Response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing workforce data..."):
                query_type = classify_query(prompt)
                
                if query_type == "irrelevant":
                    response = "I couldn't identify how this relates to our workforce data. Please ask about attendance patterns, risks, anomalies, or summaries."
                else:
                    context = _load_outputs()
                    filtered = filter_context(query_type, context)
                    response = generate_response(prompt, filtered)
                
                st.markdown(response)
        
        # Save Bot Response
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# =========================================================
# 🚀 Main Layer
# =========================================================
def render_employee_portal():
    st.markdown('<div class="header-style">👤 Employee Leave Portal</div>', unsafe_allow_html=True)
    
    st.info("Employees can submit requests here by verifying their ID and Name against our records.")
    col1, col2 = st.columns(2)
    with col1:
        emp_id = st.text_input("Employee ID (Unique)", placeholder="e.g. EMP123").strip()
    with col2:
        emp_name = st.text_input("Full Name", placeholder="e.g. Alex Doe").strip()
    
    if not emp_id or not emp_name:
        st.warning("Please enter your Employee ID and Name to continue.")
        return
        
    # Validation Logic
    valid_emps = {}
    if st.session_state.get('risk_df') is not None and "employee_id" in st.session_state.risk_df.columns:
        df = st.session_state.risk_df
        for _, row in df.iterrows():
            valid_emps[str(row['employee_id'])] = str(row.get('Emp Name', '')).strip().lower()
    else:
        # Fallback to csv if pipeline not run
        import os
        for path in ["../temp_upload.csv", "../data/raw/master_attendance.csv"]:
            full_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path)
                    emp_col = next((c for c in ['employee_id', 'Employee ID', 'Emp code'] if c in df.columns), None)
                    name_col = next((c for c in ['Emp Name', 'Employee Name'] if c in df.columns), None)
                    if emp_col:
                        for _, row in df.iterrows():
                            e_val = str(row[emp_col]).split('.')[0].strip()
                            n_val = str(row[name_col]).strip().lower() if name_col and pd.notna(row[name_col]) else ""
                            valid_emps[e_val] = n_val
                        break
                except:
                    pass
    
    if not valid_emps:
        st.error("System HR Records are empty. Please contact HR to upload the master dataset.")
        return
        
    if emp_id not in valid_emps:
        st.error(f"Invalid Info: Employee ID '{emp_id}' not found in the system.")
        return
        
    # Strict name matching (exact match ignoring case)
    sys_name = valid_emps[emp_id]
    if sys_name and emp_name.lower() != sys_name:
        st.error(f"Invalid Info: Name does not match the records for Employee ID '{emp_id}'.")
        return
        
    st.success(f"Identity Verified! Welcome, {emp_name}.")
    
    if emp_id not in st.session_state.leave_balances:
        st.session_state.leave_balances[emp_id] = 20  # Default 20 days
        save_leave_data()
        
    balance = st.session_state.leave_balances[emp_id]
    st.markdown(f"### Current Leave Balance: **{balance} days**")
    st.markdown("---")
    
    st.markdown("#### Apply for Leave")
    with st.form("leave_application_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
            
        reason = st.text_area("Reason for Leave", placeholder="Personal errand, Medical...")
        submit = st.form_submit_button("Submit Application")
        
        if submit:
            import datetime
            today = datetime.date.today()
            
            days = (end_date - start_date).days + 1
            if start_date <= today:
                st.error("Leave applications must begin from tomorrow or a future date.")
            elif days <= 0:
                st.error("End date must be on or after start date.")
            elif days > balance:
                st.error(f"Cannot apply for {days} days. You only have {balance} days left.")
            elif not reason.strip():
                st.error("Please provide a reason.")
            else:
                new_app = {
                    "Leave ID": str(uuid.uuid4())[:8],
                    "Employee ID": emp_id,
                    "Emp Name": emp_name,
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Reason": reason,
                    "Status": "Pending"
                }
                st.session_state.leave_applications = pd.concat([st.session_state.leave_applications, pd.DataFrame([new_app])], ignore_index=True)
                save_leave_data()
                st.success("Application submitted successfully!")
                st.rerun()
                
    st.markdown("---")
    st.markdown("#### Your Leave History")
    my_leaves = st.session_state.leave_applications[
        st.session_state.leave_applications["Employee ID"].astype(str) == str(emp_id)
    ]
    if my_leaves.empty:
        st.info("No leave applications found.")
    else:
        st.dataframe(my_leaves, use_container_width=True)

def render_hr_leave_management():
    st.markdown('<div class="header-style">👥 HR Leave Management</div>', unsafe_allow_html=True)
    
    pending_leaves = st.session_state.leave_applications[st.session_state.leave_applications["Status"] == "Pending"]
    
    if pending_leaves.empty:
        st.success("No pending leave applications.")
    else:
        st.markdown("#### Pending Applications")
        for idx, row in pending_leaves.iterrows():
            with st.container():
                st.markdown(f"**{row['Emp Name']} ({row['Employee ID']})**")
                days = (row['End Date'] - row['Start Date']).days + 1
                st.markdown(f"**Dates:** {row['Start Date']} to {row['End Date']} ({days} days)")
                st.markdown(f"**Reason:** {row['Reason']}")
                
                col1, col2, _ = st.columns([1, 1, 4])
                with col1:
                    if st.button("Approve", key=f"approve_{row['Leave ID']}", type="primary"):
                        st.session_state.leave_applications.at[idx, "Status"] = "Approved"
                        emp_id_str = str(row['Employee ID'])
                        st.session_state.leave_balances[emp_id_str] -= days
                        save_leave_data()
                        st.rerun()
                with col2:
                    if st.button("Reject", key=f"reject_{row['Leave ID']}"):
                        st.session_state.leave_applications.at[idx, "Status"] = "Rejected"
                        save_leave_data()
                        st.rerun()
                st.markdown("---")

    st.markdown("#### Processed Applications (Approved / Rejected)")
    processed_leaves = st.session_state.leave_applications[st.session_state.leave_applications["Status"] != "Pending"]
    if processed_leaves.empty:
        st.info("No processed applications.")
    else:
        st.dataframe(processed_leaves, use_container_width=True)

def main():
    # ---- Custom Sidebar ----
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/256/brain.png", width=80)
        st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; margin-top: 0; color: #f8fafc;'>Workforce AI</h2>", unsafe_allow_html=True)
        st.markdown("---")

        uploaded_file = st.file_uploader("Upload Attendance Dataset (CSV)", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file and not st.session_state.pipeline_done:
            temp_path = os.path.join(os.path.dirname(__file__), "..", "temp_upload.csv")
            df = pd.read_csv(uploaded_file)
            df.to_csv(temp_path, index=False)

            if st.button("Initialize Deep Analysis", use_container_width=True, type="primary"):
                f_df, p_df, r_df, ao = load_and_run_pipeline(temp_path)
                st.session_state.feature_df = f_df
                st.session_state.pattern_df = p_df
                st.session_state.risk_df = r_df
                st.session_state.agent_output = ao
                st.session_state.pipeline_done = True
                st.rerun()

        if st.session_state.pipeline_done:
            nav_options = ["Global Dashboard", "HR Copilot Assistant", "HR Leave Management"]
        else:
            nav_options = ["Welcome Hub / Upload Data", "Employee Leave Portal"]

        nav_selection = st.radio(
            "Navigation",
            options=nav_options,
            label_visibility="collapsed"
        )
            
        if st.session_state.pipeline_done:
            st.markdown("---")
            if st.button("Reset Session", use_container_width=True):
                for k in ["pipeline_done", "risk_df", "pattern_df", "feature_df", "agent_output"]:
                    st.session_state[k] = False if k == "pipeline_done" else None
                st.session_state.chat_history = [st.session_state.chat_history[0]]
                st.rerun()
            
        st.markdown("<br><br><br><br><p style='color: #64748b; font-size: 0.8rem; text-align: center;'>Agentic AI System</p>", unsafe_allow_html=True)

    # ---- App Body ----
    if nav_selection in ["Global Dashboard", "Welcome Hub / Upload Data"]:
        if not st.session_state.pipeline_done:
            st.markdown(
                """
                <div class="hero-container">
                    <h1 class="hero-title">OMNISCIENT WORKFORCE INTELLIGENCE</h1>
                    <div class="hero-subtitle">Transforming raw attendance data into strategic HR foresight. Utilize cutting-edge clustering, anomaly processing, and an integrated LLM copilot to predict employee risk before it's too late.</div>
                    <div style="display:inline-flex; align-items:center; justify-content:center; padding: 1.2rem 3rem; background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.5); border-radius: 50px; color: #ddd6fe; font-weight: 600; font-size: 1.2rem; box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3); backdrop-filter: blur(10px);">
                        Upload a dataset via sidebar to ignite the HR Engine.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        else:
            render_dashboard(
                st.session_state.risk_df,
                st.session_state.pattern_df,
                st.session_state.agent_output,
            )
    elif nav_selection == "HR Copilot Assistant":
        chat_interface()
    elif nav_selection == "Employee Leave Portal":
        render_employee_portal()
    elif nav_selection == "HR Leave Management":
        render_hr_leave_management()

if __name__ == "__main__":
    main()
