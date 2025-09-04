# src/association/apriori_rules.py

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# -------------------------------
# Config
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stroke_data_processed.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "association_rules.csv")
OUTPUT_PATH_csv =os.path.join(PROJECT_ROOT,"data","processed","simluation_transaction.csv")

TARGET = "stroke"

# -------------------------------
# Simulate transactions
# -------------------------------
def simulate_transactions(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """
    Simulate patient symptoms -> treatment transactions from stroke dataset.
    Each row becomes a 'basket' of symptoms + possible treatment.
    """
    np.random.seed(42)
    transactions = []

    for _, row in df.iterrows():
        basket = set()

        # Symptoms based on features
        if row["age"] > 60:
            basket.add("symptom:elderly")
        if row["hypertension"] == 1:
            basket.add("symptom:hypertension")
        if row["heart_disease"] == 1:
            basket.add("symptom:heart_disease")
        if row["avg_glucose_level"] > 140:
            basket.add("symptom:high_glucose")
        if row["bmi"] >= 30:
            basket.add("symptom:obese")

        # Treatments (simulate plausible mappings)
        if "symptom:hypertension" in basket:
            basket.add("treatment:antihypertensive")
        if "symptom:heart_disease" in basket:
            basket.add("treatment:cardiac_med")
        if "symptom:high_glucose" in basket:
            basket.add("treatment:insulin")
        if "symptom:obese" in basket:
            basket.add("treatment:lifestyle_change")

        # If stroke happened, add outcome tag
        if row[target_col] == 1:
            basket.add("outcome:stroke")

        transactions.append(list(basket))

    # Convert to transaction dataframe (multi-hot encoding)
    all_items = sorted({item for basket in transactions for item in basket})
    df_trans = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
    for i, basket in enumerate(transactions):
        df_trans.loc[i, basket] = 1

    return df_trans


# -------------------------------
# Mining rules
# -------------------------------
def mine_rules(df: pd.DataFrame, min_support: float = 0.05, min_conf: float = 0.6) -> pd.DataFrame:
    """
    Run Apriori + FP-Growth, return combined top-10 rules by lift.
    """
    # Apriori
    apriori_sets = apriori(df, min_support=min_support, use_colnames=True)
    apriori_rules = association_rules(apriori_sets, metric="confidence", min_threshold=min_conf)
    apriori_rules["method"] = "apriori"

    # FP-Growth
    fpg_sets = fpgrowth(df, min_support=min_support, use_colnames=True)
    fpg_rules = association_rules(fpg_sets, metric="confidence", min_threshold=min_conf)
    fpg_rules["method"] = "fpgrowth"

    rules = pd.concat([apriori_rules, fpg_rules], ignore_index=True)

    # Keep top 10 by lift then confidence
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False).head(10).reset_index(drop=True)
    return rules

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("[INFO] Simulating transactions...")
    trans_df = simulate_transactions(df)
    trans_df.to_csv(OUTPUT_PATH_csv, index=False)

    print("[INFO] Mining rules...")
    rules = mine_rules(trans_df, min_support=0.05, min_conf=0.6)

    # Convert frozensets to clean strings
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

    print("\n[TOP 10 RULES]")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift", "method"]])

    rules.to_csv(OUTPUT_PATH, index=False)
    print(f"[SAVE] Association rules written to {OUTPUT_PATH}")
