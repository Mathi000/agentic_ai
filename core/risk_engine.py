import pandas as pd
import joblib


# =========================================================
# 📦 Load Model
# =========================================================
def load_model(model_path: str):
    return joblib.load(model_path)


# =========================================================
# 🔮 Predict Risk
# =========================================================
def predict_risk(feature_df: pd.DataFrame) -> pd.DataFrame:
    print("📥 Loading pattern data...")

    pattern_df = pd.read_csv("data/pattern_output.csv")

    # Merge Dev1 + Dev2 outputs
    df = feature_df.merge(pattern_df, on="employee_id")

    employee_ids = df["employee_id"]

    # Prepare features
    X = df.drop(columns=["employee_id"])
    X = pd.get_dummies(X)

    print("📦 Loading trained model...")
    model = load_model("models/risk_model.pkl")

    # Ensure same columns as training
    try:
        model_features = model.get_booster().feature_names
        X = X.reindex(columns=model_features, fill_value=0)
    except:
        pass

    print("🔮 Predicting risk...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # =====================================================
    # ✅ FIX: TRUE RISK SCORE (High Risk Probability)
    # =====================================================
    if probabilities.shape[1] >= 3:
        # Column 2 = High risk probability
        risk_scores = probabilities[:, 2]
    else:
        # Fallback (if only 2 classes)
        risk_scores = probabilities.max(axis=1)

    # Map labels
    risk_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    risk_categories = [risk_map.get(p, "Unknown") for p in predictions]

    # Base output
    output = pd.DataFrame({
        "employee_id": employee_ids,
        "risk_score": risk_scores,
        "risk_category": risk_categories
    })

    # Attach latest meta info for better analysis (department, name, date)
    meta_cols = [c for c in ["employee_id", "Emp Name", "department", "date"] if c in feature_df.columns]
    if meta_cols:
        meta = feature_df[meta_cols].copy()
        if "date" in meta.columns:
            meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
            meta = meta.sort_values("date")
        meta = meta.drop_duplicates(subset=["employee_id"], keep="last")
        output = output.merge(meta, on="employee_id", how="left")

    output.to_csv("data/risk_output.csv", index=False)

    print("✅ risk_output.csv generated successfully!")

    return output


# =========================================================
# 🧪 TEST RUN
# =========================================================
if __name__ == "__main__":
    print("🚀 Running Risk Engine...")

    feature_df = pd.read_csv("data/processed/feature_engineered.csv")

    result = predict_risk(feature_df)

    print("\n📊 Sample Output:")
    print(result.head())
