import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

print("Loading dataset...")

df = pd.read_csv("data/processed/feature_engineered.csv")

# Behavioral features
features = [
    "working_hours",
    "late_minutes",
    "checkin_minutes",
    "checkin_variance",
    "overtime_flag",
    "late_flag",
    "late_ratio_7d",
    "consecutive_work_days",
    "days_present_30d",
    "absence_ratio_30d"
]

feature_df = df[features]

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df)

print("Running clustering...")

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster label
df["cluster_label"] = clusters

print("\nCluster Distribution:")
print(df["cluster_label"].value_counts())

# Map clusters to behavior labels
cluster_map = {
    0: "Stable",
    1: "Irregular",
    2: "High Deviation"
}

df["cluster_label"] = df["cluster_label"].map(cluster_map)

print("\nBehavior Distribution:")
print(df["cluster_label"].value_counts())

print("\nRunning anomaly detection...")

# Train Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)

anomaly_scores = iso.fit_predict(scaled_features)

# Convert output (-1 anomaly, 1 normal)
df["anomaly_flag"] = anomaly_scores

# Convert to readable labels
df["anomaly_flag"] = df["anomaly_flag"].map({
    1: "Normal",
    -1: "Anomaly"
})

print("\nAnomaly Detection Results:")
print(df["anomaly_flag"].value_counts())
print("\nRunning trend detection...")

def detect_trend(row):
    if row["late_ratio_7d"] > 0.4:
        return "Increasing Lateness"
    elif row["checkin_variance"] > 60:
        return "Unstable Checkin"
    elif row["absence_ratio_30d"] > 0.3:
        return "High Absence Trend"
    else:
        return "Stable Trend"

df["trend_flag"] = df.apply(detect_trend, axis=1)

print("\nTrend Detection Results:")
print(df["trend_flag"].value_counts())
print("\nGenerating pattern_output.csv...")

output_cols = [
    "employee_id",
    "Emp Name",
    "department",
    "date",
    "cluster_label",
    "anomaly_flag",
    "trend_flag",
]

# Keep only columns that actually exist
output_cols = [c for c in output_cols if c in df.columns]
output = df[output_cols]

output.to_csv("data/pattern_output.csv", index=False)

print("pattern_output.csv generated successfully!")
import matplotlib.pyplot as plt

print("\nGenerating behavior graph...")

behavior_counts = df["cluster_label"].value_counts()

plt.figure()
behavior_counts.plot(kind="bar")

plt.title("Employee Behavior Clustering")
plt.xlabel("Behavior Type")
plt.ylabel("Number of Employees")

plt.show()
print("\nGenerating anomaly graph...")

anomaly_counts = df["anomaly_flag"].value_counts()

plt.figure()
anomaly_counts.plot(kind="bar")

plt.title("Anomaly Detection Results")
plt.xlabel("Employee Type")
plt.ylabel("Count")

plt.show()
print("\nGenerating trend graph...")

trend_counts = df["trend_flag"].value_counts()

plt.figure()
trend_counts.plot(kind="bar")

plt.title("Attendance Trend Detection")
plt.xlabel("Trend Type")
plt.ylabel("Number of Employees")

plt.show()
