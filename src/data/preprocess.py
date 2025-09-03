import pandas as pd
import os

from load import load_csv

# Load raw data
df = load_csv(os.path.join("data", "raw", "stroke_data.csv"))
print(f"Raw data shape: {df.shape}")

# =========================
# Preprocessing
# =========================

df_processed = df.copy()

# Fill missing numeric values with median
for col in ['bmi', 'avg_glucose_level', 'age']:
    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

# Fill missing categorical values with mode
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

# One-hot encode categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols)

print(f"Processed data shape: {df_processed.shape}")
print(f"Columns: {df_processed.columns.tolist()}")

# =========================
# Save processed data
# =========================
processed_folder = os.path.join("data", "processed")
os.makedirs(processed_folder, exist_ok=True)  # ensure folder exists

processed_path = os.path.join(processed_folder, "stroke_data_processed.csv")
df_processed.to_csv(processed_path, index=False)

print(f"Processed data saved to: {processed_path}")
