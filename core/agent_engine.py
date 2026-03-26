import pandas as pd
from typing import Dict, Any, List


# =========================================================
# 🚨 Alert Generation
# =========================================================
def generate_alerts(risk_df: pd.DataFrame) -> List[str]:
    """
    Generate HR alerts from risk prediction output.
    Returns a list of alert strings.
    """
    alerts = []

    # High-risk employees
    high_risk = risk_df[risk_df["risk_category"] == "High"]
    if not high_risk.empty:
        count = len(high_risk)
        alerts.append(
            f"🔴 {count} employee(s) flagged as HIGH risk — immediate review recommended."
        )

        # Department-level alerts
        if "department" in high_risk.columns:
            dept_counts = high_risk["department"].value_counts()
            for dept, cnt in dept_counts.head(3).items():
                alerts.append(
                    f"⚠️ Department '{dept}' has {cnt} high-risk employee(s)."
                )

    # Medium risk
    medium_risk = risk_df[risk_df["risk_category"] == "Medium"]
    if not medium_risk.empty:
        alerts.append(
            f"🟡 {len(medium_risk)} employee(s) in MEDIUM risk — monitor closely."
        )

    # Very high scores
    if "risk_score" in risk_df.columns:
        critical = risk_df[risk_df["risk_score"] > 0.9]
        if not critical.empty:
            names = critical["Emp Name"].tolist() if "Emp Name" in critical.columns else []
            if names:
                top = ", ".join(names[:5])
                alerts.append(
                    f"🚨 Critical risk scores (>0.9) for: {top}"
                )

    if not alerts:
        alerts.append("✅ No critical attendance alerts at this time.")

    return alerts


# =========================================================
# 💡 Recommendation Generation
# =========================================================
def generate_recommendations(risk_df: pd.DataFrame) -> List[str]:
    """
    Generate HR action recommendations based on risk analysis.
    """
    recs = []

    high_risk = risk_df[risk_df["risk_category"] == "High"]

    if not high_risk.empty:
        recs.append(
            "📋 Schedule one-on-one meetings with high-risk employees to understand concerns."
        )
        recs.append(
            "📊 Review workload distribution for departments with multiple high-risk flags."
        )

        if "department" in high_risk.columns:
            top_dept = high_risk["department"].value_counts().idxmax()
            recs.append(
                f"🔍 Conduct a focused review of '{top_dept}' — highest concentration of risk."
            )

    medium_risk = risk_df[risk_df["risk_category"] == "Medium"]
    if not medium_risk.empty:
        recs.append(
            "📅 Set up bi-weekly check-ins with medium-risk employees to prevent escalation."
        )

    total = len(risk_df)
    low_risk = risk_df[risk_df["risk_category"] == "Low"]
    if len(low_risk) < total * 0.5:
        recs.append(
            "⚡ Overall risk levels are elevated — consider team-wide wellness initiatives."
        )
    else:
        recs.append(
            "✅ Majority of employees are low-risk — maintain current engagement practices."
        )

    return recs


# =========================================================
# 📝 Summary Generation
# =========================================================
def generate_summaries(risk_df: pd.DataFrame) -> List[str]:
    """
    Generate overall workforce summaries.
    """
    summaries = []
    total = len(risk_df)

    for cat in ["Low", "Medium", "High"]:
        count = len(risk_df[risk_df["risk_category"] == cat])
        pct = round(count / total * 100, 1) if total else 0
        summaries.append(f"{cat} Risk: {count} employees ({pct}%)")

    if "risk_score" in risk_df.columns:
        avg = round(risk_df["risk_score"].mean(), 3)
        summaries.append(f"Average risk score: {avg}")

    return summaries


# =========================================================
# 🤖 Main Agent Layer
# =========================================================
def run_agent_layer(risk_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the full agent intelligence layer.
    Returns dict with alerts, recommendations, and summaries.
    """
    return {
        "alerts": generate_alerts(risk_df),
        "recommendations": generate_recommendations(risk_df),
        "summaries": generate_summaries(risk_df),
    }


# =========================================================
# 🧪 TEST
# =========================================================
if __name__ == "__main__":
    df = pd.read_csv("data/risk_output.csv")
    output = run_agent_layer(df)

    print("\n🚨 Alerts:")
    for a in output["alerts"]:
        print(f"  {a}")

    print("\n💡 Recommendations:")
    for r in output["recommendations"]:
        print(f"  {r}")

    print("\n📝 Summaries:")
    for s in output["summaries"]:
        print(f"  {s}")
