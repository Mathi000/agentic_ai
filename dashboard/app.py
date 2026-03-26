import sys
import os

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
    /* ---- Fonts & Global Aesthetics ---- */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .block-container { 
        padding-top: 1.5rem !important; 
        max-width: 1400px; 
    }

    /* ---- Stunning Animated Landing Page ---- */
    .hero-container {
        text-align: center; 
        padding: 5rem 2rem; 
        background: radial-gradient(circle at center, rgba(139, 92, 246, 0.15) 0%, transparent 60%); 
        border-radius: 40px; 
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: conic-gradient(from 0deg at 50% 50%, rgba(139, 92, 246, 0.1), transparent 60%);
        animation: rotate 20s linear infinite;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }

    .hero-title {
        font-size: 4.5rem; 
        font-weight: 800; 
        letter-spacing: -2px; 
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem; 
        color: #94a3b8; 
        max-width: 700px; 
        margin: 0 auto 3rem auto; 
        line-height: 1.6; 
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* ---- KPI Cards ---- */
    .kpi-wrapper {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .kpi-wrapper::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent);
    }
    .kpi-wrapper:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px -10px rgba(139, 92, 246, 0.2);
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .kpi-value { 
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0; 
        font-size: 3.5rem; 
        font-weight: 800;
        line-height: 1;
        letter-spacing: -2px;
    }
    .kpi-label { 
        color: #94a3b8; 
        margin-top: 1rem; 
        font-size: 0.9rem; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        font-weight: 600;
    }

    /* ---- Section Headers ---- */
    .header-style {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* ---- Insight Cards ---- */
    .insight-box {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        font-size: 1rem;
        line-height: 1.5;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        color: #e2e8f0;
        position: relative;
        overflow: hidden;
    }
    .insight-box::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
    }
    .insight-box:hover {
        background: rgba(30, 41, 59, 0.8);
        transform: translateX(6px);
    }
    .insight-alert::before { background: #ef4444; }
    .insight-rec::before { background: #10b981; }

    /* ---- Streamlit Component Overrides ---- */
    div[data-baseweb="select"] > div {
        border-radius: 12px;
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.05);
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
    total = len(risk_df)
    high_count = len(risk_df[risk_df["risk_category"] == "High"])
    anomaly_count = len(pattern_df[pattern_df["anomaly_flag"] == "Anomaly"]) if "anomaly_flag" in pattern_df.columns else 0
    avg_risk = round(risk_df["risk_score"].mean(), 3) if "risk_score" in risk_df.columns else 0

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
def main():
    # ---- Custom Sidebar ----
    with st.sidebar:
        st.image("https://img.icons8.com/parakeet-line/144/8b5cf6/artificial-intelligence.png", width=70)
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
            st.success("Analysis Engine Online")
            nav_selection = st.radio(
                "Navigation",
                options=["Global Dashboard", "HR Copilot Assistant"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            if st.button("Reset Session", use_container_width=True):
                for k in ["pipeline_done", "risk_df", "pattern_df", "feature_df", "agent_output"]:
                    st.session_state[k] = False if k == "pipeline_done" else None
                st.session_state.chat_history = [st.session_state.chat_history[0]]
                st.rerun()
        else:
            nav_selection = None
            
        st.markdown("<br><br><br><br><p style='color: #64748b; font-size: 0.8rem; text-align: center;'>Agentic AI System</p>", unsafe_allow_html=True)

    # ---- App Body ----
    if not st.session_state.pipeline_done:
        st.markdown(
            """
            <div class="hero-container">
                <h1 class="hero-title">Omniscient Workforce Intelligence</h1>
                <p class="hero-subtitle">
                    Transforming raw attendance data into strategic HR foresight. Utilize cutting-edge clustering, anomaly processing, and an integrated LLM copilot to predict employee risk before it's too late.
                </p>
                <div style="display:inline-flex; align-items:center; justify-content:center; padding: 1.2rem 3rem; background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.5); border-radius: 50px; color: #ddd6fe; font-weight: 600; font-size: 1.2rem; box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3); backdrop-filter: blur(10px);">
                    Upload a dataset to ignite the engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Routing based on sidebar radio
    if nav_selection == "Global Dashboard":
        render_dashboard(
            st.session_state.risk_df,
            st.session_state.pattern_df,
            st.session_state.agent_output,
        )
    else:
        chat_interface()

if __name__ == "__main__":
    main()
