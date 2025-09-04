# src/unsupervised/clustering.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

# -------------------------------
# Config
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stroke_data_processed.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "cluster_profiles.md")
MODEL_PATH = os.path.join(PROJECT_ROOT,"models","cluster_model.pkl")

TARGET = "stroke"

# -------------------------------
# Utility function
# -------------------------------
def run_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, float]:
    features = df.drop(columns=[c for c in [TARGET, "id"] if c in df.columns], errors="ignore")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    df_kmeans = df.copy()
    df_kmeans["cluster"] = labels

    # Map numeric clusters -> risk names (adjust if you analyze differently)
    cluster_name_map = {
        0: "High Risk",
        1: "Moderate Risk",
        2: "Low Risk"
    }
    df_kmeans["cluster_name"] = df_kmeans["cluster"].map(cluster_name_map)
    # Save the model + scaler + feature order
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": kmeans, "scaler": scaler, "features": list(features.columns)}, MODEL_PATH)

    # Save the scaler
    joblib.dump(scaler, 'C:\\Users\\moham\\OneDrive\\Desktop\\ML_Learnings\\ai-healthcare-system\\models\\scaler.pkl')

    # Save the kmeans model
    joblib.dump(kmeans, 'C:\\Users\\moham\\OneDrive\\Desktop\\ML_Learnings\\ai-healthcare-system\\models\\kmeans.pkl')
    print(f"[SAVE] KMeans model and scaler saved at {MODEL_PATH}")

    score = silhouette_score(scaled, labels)
    return df_kmeans, score


def run_dbscan(df: pd.DataFrame, eps: float = 1.5, min_samples: int = 50) -> tuple[pd.DataFrame, float | None]:
    features = df.drop(columns=[c for c in [TARGET, "id"] if c in df.columns], errors="ignore")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled)

    df_dbscan = df.copy()
    df_dbscan["cluster"] = labels

    score = None
    if len(set(labels)) > 1 and -1 not in set(labels):  # skip if all noise
        score = silhouette_score(scaled, labels)
    return df_dbscan, score


def profile_clusters(df: pd.DataFrame, method: str, score: float | None) -> str:
    """Generate a markdown cluster profile summary with human-readable names if available."""
    summary = [f"# Cluster Profiles - {method}\n"]
    if score:
        summary.append(f"- Silhouette Score: **{score:.4f}**\n")
    else:
        summary.append("- Silhouette Score: N/A (only one cluster or noise)\n")

    summary.append("## Cluster Sizes\n")
    if "cluster_name" in df.columns:
        cluster_sizes = df["cluster_name"].value_counts().to_frame().reset_index()
        cluster_sizes.columns = ["cluster_name", "count"]
        summary.append(cluster_sizes.to_markdown(index=False))
    else:
        summary.append(df["cluster"].value_counts().sort_index().to_markdown())

    summary.append("\n## Mean Feature Values by Cluster\n")
    if "cluster_name" in df.columns:
        profile = df.groupby("cluster_name").mean(numeric_only=True)
    else:
        profile = df.groupby("cluster").mean(numeric_only=True)
    summary.append(profile.to_markdown())

    return "\n".join(summary)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # --- Run KMeans ---
    clustered_kmeans, kmeans_score = run_kmeans(df, n_clusters=3)
    print(f"[KMeans] Silhouette Score: {kmeans_score:.4f}")
    print(clustered_kmeans["cluster_name"].value_counts())

    # --- Run DBSCAN ---
    clustered_dbscan, dbscan_score = run_dbscan(df, eps=1.5, min_samples=50)
    print(f"[DBSCAN] Silhouette Score: {dbscan_score}")
    print(clustered_dbscan["cluster"].value_counts())

    # --- Save profiles to markdown ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    md_content = []
    md_content.append(profile_clusters(clustered_kmeans, "KMeans", kmeans_score))
    md_content.append("\n---\n")
    md_content.append(profile_clusters(clustered_dbscan, "DBSCAN", dbscan_score))

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(md_content))

    print(f"[SAVE] Cluster profiles written to {OUTPUT_PATH}")
