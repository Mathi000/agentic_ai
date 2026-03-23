import pandas as pd
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import textwrap


# =========================================================
# 🎯 FINAL ROBUST RISK LOGIC (RANK-BASED)
# =========================================================
def create_risk_labels(df: pd.DataFrame) -> pd.Series:
    scores = []

    for _, row in df.iterrows():
        score = 0

        # 🔴 Strong indicators
        if row["anomaly_flag"] == "Anomaly":
            score += 3
        if row["cluster_label"] == "High Deviation":
            score += 2
        if row["trend_flag"] == "High Absence Trend":
            score += 2

        # 🟡 Medium indicators
        if row["cluster_label"] == "Irregular":
            score += 1
        if row["trend_flag"] == "Increasing Lateness":
            score += 1

        scores.append(score)

    scores = pd.Series(scores)

    # =====================================================
    # ✅ RANK-BASED CLASSIFICATION (NO COLLAPSE)
    # =====================================================
    ranks = scores.rank(method="first")
    n = len(scores)

    labels = []

    for r in ranks:
        if r > 0.66 * n:
            labels.append(2)  # High
        elif r > 0.33 * n:
            labels.append(1)  # Medium
        else:
            labels.append(0)  # Low

    return pd.Series(labels)


# =========================================================
# 🚀 MODEL TRAINING
# =========================================================
def train_risk_model(X: pd.DataFrame, y: pd.Series):
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X, y)

    joblib.dump(model, "models/risk_model.pkl")
    print("✅ Model saved to models/risk_model.pkl")

    return model


# =========================================================
# 📊 FEATURE IMPORTANCE (VERTICAL CLEAN GRAPH)
# =========================================================
def plot_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(10)

    # Clean feature names
    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("_", " ")
        .str.title()
    )

    # Wrap long labels
    importance_df["feature"] = importance_df["feature"].apply(
        lambda x: "\n".join(textwrap.wrap(x, width=12))
    )

    plt.figure(figsize=(12, 6))
    plt.bar(importance_df["feature"], importance_df["importance"])

    plt.xticks(rotation=0, ha="center")

    plt.title("Top Features Influencing Employee Risk")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# =========================================================
# 🧪 MAIN PIPELINE
# =========================================================
if __name__ == "__main__":
    print("📥 Loading datasets...")

    feature_df = pd.read_csv("data/processed/feature_engineered.csv")
    pattern_df = pd.read_csv("data/pattern_output.csv")

    df = feature_df.merge(pattern_df, on="employee_id")

    print("🧠 Creating risk labels...")
    y = create_risk_labels(df)

    # ✅ Check distribution (must have all 3 classes)
    print("\n📊 Label Distribution:")
    print(y.value_counts())

    # Ensure labels are 0,1,2 for XGBoost
    unique_classes = sorted(y.unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    y = y.map(label_mapping)

    print("\nLabel mapping:", label_mapping)

    # Prepare features
    X = df.drop(columns=["employee_id"])
    X = pd.get_dummies(X)

    print("🚀 Training XGBoost model...")
    model = train_risk_model(X, y)

    print("📊 Generating feature importance...")
    plot_feature_importance(model, X)

    print("\n✅ Training pipeline completed successfully!")