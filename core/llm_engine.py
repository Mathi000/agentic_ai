import os
import json
import hashlib
from functools import lru_cache
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =========================================================
# LLM Agent Layer (Groq)
# =========================================================
# Expect API key in env var: GROQ_API_KEY
# Example (PowerShell):
#   $env:GROQ_API_KEY="your_key_here"


SYSTEM_PROMPT = "Answer briefly and directly. Max 5 lines. Ignore unrelated info."
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = 100
TEMPERATURE = 0


def _load_outputs() -> Dict[str, Any]:
    """
    Load pattern_output.csv and risk_output.csv for fallback context.
    Returns a dict with patterns and anomalies arrays plus a summary object.
    """
    context: Dict[str, Any] = {"patterns": [], "anomalies": [], "summary": {}}

    try:
        pattern_df = pd.read_csv("data/pattern_output.csv")
        if not pattern_df.empty:
            context["patterns"] = pattern_df.head(50).to_dict(orient="records")
            context["anomalies"] = (
                pattern_df[pattern_df["anomaly_flag"] == "Anomaly"]
                .head(50)
                .to_dict(orient="records")
            )
            context["summary"]["pattern_counts"] = (
                pattern_df["cluster_label"].value_counts().to_dict()
            )
            context["summary"]["anomaly_counts"] = (
                pattern_df["anomaly_flag"].value_counts().to_dict()
            )
            context["summary"]["trend_counts"] = (
                pattern_df["trend_flag"].value_counts().to_dict()
            )
    except Exception:
        pass

    try:
        risk_df = pd.read_csv("data/risk_output.csv")
        if not risk_df.empty:
            context["summary"]["risk_counts"] = (
                risk_df["risk_category"].value_counts().to_dict()
            )
            if "department" in risk_df.columns:
                # Counts of high risk per department
                high_counts = (
                    risk_df[risk_df["risk_category"] == "High"]
                    .groupby("department")["employee_id"]
                    .nunique()
                )
                total_counts = (
                    risk_df.groupby("department")["employee_id"]
                    .nunique()
                )

                dept_percent = (
                    (high_counts / total_counts.replace(0, pd.NA)) * 100
                ).dropna().sort_values(ascending=False)

                context["summary"]["high_risk_by_department"] = (
                    high_counts.sort_values(ascending=False).head(10).to_dict()
                )
                context["summary"]["high_risk_pct_by_department"] = (
                    dept_percent.head(10).round(2).to_dict()
                )
    except Exception:
        pass

    return context


def classify_query(query: str) -> str:
    """
    Classify query type: pattern / anomaly / summary / irrelevant
    """
    q = (query or "").lower()
    if not q.strip():
        return "irrelevant"

    pattern_kw = ["pattern", "behavior", "cluster", "trend", "late", "overtime"]
    anomaly_kw = ["anomaly", "unusual", "outlier", "spike", "abnormal"]
    summary_kw = ["summary", "overview", "stats", "distribution", "count", "risk"]

    if any(k in q for k in anomaly_kw):
        return "anomaly"
    if any(k in q for k in pattern_kw):
        return "pattern"
    if any(k in q for k in summary_kw):
        return "summary"

    return "irrelevant"


def filter_context(query_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter context by query type and keep it compact.
    """
    if not context:
        context = _load_outputs()

    if query_type == "pattern":
        return {"patterns": context.get("patterns", [])[:30]}
    if query_type == "anomaly":
        return {"anomalies": context.get("anomalies", [])[:30]}
    if query_type == "summary":
        return {"summary": context.get("summary", {})}

    return {}


def _hash_payload(query: str, filtered_context: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"q": query, "ctx": filtered_context}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=256)
def _cached_llm_response(payload_hash: str, query: str, filtered_context_json: str) -> str:
    try:
        from groq import Groq
    except Exception:
        return "⚠️ The `groq` package is missing. Run `pip install groq`."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing! Please create a `.env` file in your `agentic_ai` folder and add your API key like: `GROQ_API_KEY=your-api-key-here`"

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Query: {query}\nContext: {filtered_context_json}",
        },
    ]

    debug = os.getenv("GROQ_DEBUG") == "1"
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ Groq API Error: {exc} — Please check your API key and network connection."


def generate_response(query: str, filtered_context: Dict[str, Any]) -> str:
    """
    Call Groq API with compact context. Cached for frequent queries.
    """
    filtered_context_json = json.dumps(filtered_context, default=str)
    payload_hash = _hash_payload(query, filtered_context)
    return _cached_llm_response(payload_hash, query, filtered_context_json)


def main(query: str, context: Dict[str, Any]) -> str:
    """
    Pipeline: classify -> filter -> generate
    """
    query_type = classify_query(query)

    if query_type == "irrelevant":
        return "This query is not related to attendance analysis."

    filtered_context = filter_context(query_type, context or {})

    return generate_response(query, filtered_context)
