# src/app/streamlit_app.py

from __future__ import annotations
import os
import ast
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
MODEL_PATH = ROOT / "models" / "model.pkl"
CLUSTER_MODEL_PATH = ROOT / "models" / "cluster_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "stroke_data_processed.csv"
RULES_PATH = ROOT / "association_rules.csv"
CLUSTER_PROFILES_PATH = ROOT / "cluster_profiles.md"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
KMEANS_PATH = ROOT / "models" / "kmeans.pkl"

# -------------------------------
# Load utilities
# -------------------------------
@st.cache_resource
def load_model():
    """Load the supervised model pipeline."""
    if not MODEL_PATH.exists():
        st.error("Supervised model not found. Please train the model in Milestone 2.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    """Load the processed dataset."""
    if not DATA_PATH.exists():
        st.error("Processed data not found. Please run preprocessing first.")
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_rules():
    """Load and parse association rules."""
    if not RULES_PATH.exists():
        st.warning("Association rules file not found. Please run Milestone 4 first.")
        return None
    rules = pd.read_csv(RULES_PATH)

    # Convert comma-separated strings → sets
    def to_set(s):
        return set([x.strip() for x in str(s).split(",") if x.strip()])

    rules["antecedents"] = rules["antecedents"].apply(to_set)
    rules["consequents"] = rules["consequents"].apply(to_set)

    return rules



@st.cache_resource
def load_cluster_model():
    """Load the clustering model and scaler."""
    if not CLUSTER_MODEL_PATH.exists():
        st.error("Clustering model not found. Please train and save it in Milestone 3.")
        return None
    return joblib.load(CLUSTER_MODEL_PATH)

# Updated load_cluster_profiles function in src/app/streamlit_app.py
# Updated load_cluster_profiles function in src/app/streamlit_app.py
@st.cache_data
def load_cluster_profiles():
    """Loads and returns the cluster profiles markdown content."""
    # Assuming the path is correctly defined at the top of your script
    if not CLUSTER_PROFILES_PATH.exists():
        st.warning("Cluster profiles not found. Please create 'cluster_profiles.md' in Milestone 3.")
        return {}
    
    profiles = {}
    current_cluster_id = None
    
    with open(CLUSTER_PROFILES_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split the content by markdown header for each cluster
    sections = content.split("###")
    
    for section in sections:
        section = section.strip()
        if section.startswith("Cluster"):
            # The header is something like 'Cluster 0: High-Risk Group'
            header_line = section.split("\n")[0]
            try:
                # Extract the cluster number from the header
                cluster_id = int(header_line.split(":")[0].strip().split(" ")[1])
                # The profile text is everything after the header
                profile_text = "\n".join(section.split("\n")[1:]).strip()
                profiles[cluster_id] = profile_text
            except (IndexError, ValueError):
                continue
    
    return profiles
# Add this helper function to your script's loading section
@st.cache_data
def get_cluster_feature_names():
    # Load the processed data to get the original feature names
    processed_df = pd.read_csv(Path(r"C:\Users\moham\OneDrive\Desktop\ML_Learnings\ai-healthcare-system\data\processed\stroke_data_processed.csv"))
    # Drop the target and ID to get the features used for clustering
    cluster_features = processed_df.drop(columns=['stroke', 'id'], errors='ignore')
    return list(cluster_features.columns)

@st.cache_resource
def load_cluster_assets():
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return scaler, kmeans

CLUSTER_NAME_MAP = {
    0: "High-Risk Group",
    1: "Moderate-Risk Group",
    2: "Low-Risk Younger Group"
}


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="AI Healthcare App", layout="wide")
st.title("🩺 AI-Powered Healthcare System")

model = load_model()
df = load_data()
rules = load_rules()
# cluster_profiles = load_cluster_profiles()
# scaler, kmeans = load_cluster_assets()
# Note: your original code loaded `kmeans, scaler, cluster_cols`.
# This assumes you saved them together. I've updated load_cluster_model to reflect this.
# If you saved them separately, adjust accordingly.
if CLUSTER_MODEL_PATH.exists():
     scaler,kmeans= load_cluster_assets()
     cluster_profiles = load_cluster_profiles()
     cluster_cols = get_cluster_feature_names()
else:
    kmeans, scaler,cluster_cols= None, None, None
    cluster_profiles = {}

if not all([model, df is not None and not df.empty, kmeans, scaler, cluster_cols, rules is not None and not rules.empty]):


# -------------------------------
# Input Form
# -------------------------------
 st.subheader("Single Patient Prediction")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        avg_glucose = st.number_input("Avg Glucose Level", min_value=50.0, value=100.0)
    
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, value=25.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    with col3:
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.3, 0.01)

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Display Results
# -------------------------------
if submitted:
    st.markdown("---")
    st.header("Prediction & Recommendations")
    # Create a dictionary for the patient's data
    patient_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }
        
    # 1. Prepare DataFrame for model input
    patient = pd.DataFrame([patient_data])
        # Correctly encode categorical features
    # This should match the categories seen during model training
    patient_encoded = pd.get_dummies(patient, columns=[
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ], drop_first=False)

    all_training_cols = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Female', 'gender_Male', 'gender_Other',
        'ever_married_No', 'ever_married_Yes',
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_Unknown'
    ]

    patient_final = patient_encoded.reindex(columns=all_training_cols, fill_value=0)

    # --- Supervised Prediction
    proba = model.predict_proba(patient_final)[:, 1][0]
    pred = int(proba >= threshold)

    if pred == 1:
        st.error(f"**Diagnosis Prediction:** HIGH RISK (Probability: {proba:.3f})")
    else:
        st.success(f"**Diagnosis Prediction:** LOW RISK (Probability: {proba:.3f})")

    st.markdown("---")

    # --- Unsupervised Cluster
   
   # --- Unsupervised Cluster
    st.subheader("Patient Risk Cluster")
    try:
        # Prepare patient data for clustering
        # This ensures the input DataFrame has all the required columns
        patient_for_clustering = patient.reindex(columns=cluster_cols, fill_value=0)
        
        # Transform the patient data using the loaded scaler
        row_scaled = scaler.transform(patient_for_clustering)
        scaler, kmeans = load_cluster_assets()
        
        # Predict the cluster
        cluster = int(kmeans.predict(row_scaled)[0])
        cluster_name = CLUSTER_NAME_MAP.get(cluster, f"Cluster {cluster} (Name Not Found)")
        st.info(f"**Cluster ID:** `{cluster}` | **Profile:** `{cluster_name}`")
        if cluster in cluster_profiles:
            st.write(cluster_profiles[cluster])
        else:
            st.warning("Cluster profile not found for this ID.")
    except Exception as e:
        st.error(f"Error predicting cluster: {e}. Please ensure the cluster model is compatible.")
    st.markdown("---")
    # --- Association Rules
    st.subheader("Recommended Treatments & Diagnoses")
    if rules is not None and not rules.empty:
        # Build transaction based on patient's features
        items = set()
        if age >= 65: items.add("symptom:elderly")
        if hypertension == 1: items.add("symptom:hypertension")
        if heart_disease == 1: items.add("symptom:heart_disease")
        if avg_glucose > 140: items.add("symptom:high_glucose")
        if bmi >= 30: items.add("symptom:obese")

        recs = []
        for _, r in rules.iterrows():
            if r["antecedents"].issubset(items):
                recs.append((r["antecedents"], r["consequents"], r["lift"], r["confidence"]))

        recs = sorted(recs, key=lambda x: (x[2], x[3]), reverse=True)[:5]

        if recs:
            for ant, cons, lift, conf in recs:
                st.write(
                    f"**{', '.join(ant)}** → **{', '.join(cons)}** "
                    f"(Lift: {lift:.2f}, Confidence: {conf:.2f})"
                )
        else:
            st.info("No matching rules for this patient.")
    else:
        st.warning("Association rules could not be loaded.")
