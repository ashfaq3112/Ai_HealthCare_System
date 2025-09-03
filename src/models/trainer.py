# src/models/trainer.py

"""
Trainer script:
 - Loads processed data
 - Builds preprocessing + pipelines from src.models.baseline
 - Runs 5-fold stratified CV (ROC AUC) per pipeline
 - Picks best model by mean CV ROC AUC
 - Retrains best model on full train set, evaluates on test set
 - Saves best model to models/model.pkl
 - Optionally: runs SHAP for XGBoost and prints top features
"""
from __future__ import annotations
import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# ensure project root is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline import build_preprocessor, build_pipelines

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve
)
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stroke_data_processed.csv")
TARGET = "stroke"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RANDOM_STATE = 42
N_SPLITS = 5

# ----------------------------
# Utility: CV AUC (stratified folds)
# ----------------------------
def cv_auc_scores(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> Tuple[float, list]:
    """Runs stratified cross-validation and returns mean ROC AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        pipeline.fit(X_tr, y_tr)
        probs = pipeline.predict_proba(X_va)[:, 1]
        aucs.append(float(roc_auc_score(y_va, probs)))
    return float(np.mean(aucs)), aucs

# ----------------------------
# Utility: get feature names after preprocessor
# ----------------------------
def get_feature_names(preprocessor, X_orig: pd.DataFrame):
    """Safely retrieves feature names after the preprocessor transformation."""
    try:
        return preprocessor.get_feature_names_out(X_orig.columns).tolist()
    except Exception:
        try:
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            return [c for c in X_orig.columns if c not in ("id", TARGET)]

# ----------------------------
# Main train + eval
# ----------------------------
def train_and_eval() -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at: {DATA_PATH}\nRun preprocess.py first.")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Found: {df.columns.tolist()}")

    # Split features / target
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df.columns], errors="ignore")
    y = df[TARGET].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[DATA] Train: {X_train.shape}, Test: {X_test.shape}")

    # Preprocessor + pipelines
    preprocessor, num_cols, cat_cols = build_preprocessor(df, target=TARGET)
    # 💥 CRITICAL UPDATE: Pass y_train to build_pipelines
    pipelines = build_pipelines(preprocessor, y_train, random_state=RANDOM_STATE)
    print(f"[MODELS] Pipelines created: {list(pipelines.keys())}")

    # Cross-validation
    cv_results = {}
    print("\n[CV] Running stratified 5-fold CV (ROC AUC)")
    for name, pipe in pipelines.items():
        # 💥 CRITICAL UPDATE: Pass X_train and y_train to cv_auc_scores
        mean_auc, fold_aucs = cv_auc_scores(pipe, X_train, y_train, n_splits=N_SPLITS)
        cv_results[name] = {"mean_auc": mean_auc, "fold_aucs": fold_aucs}
        print(f"  - {name}: mean_auc={mean_auc:.4f} | folds: {', '.join(f'{a:.4f}' for a in fold_aucs)}")

    # Best model
    best_name = max(cv_results, key=lambda n: cv_results[n]["mean_auc"])
    best_pipeline = pipelines[best_name]
    print(f"\n[SELECT] Best model: {best_name} (mean AUC={cv_results[best_name]['mean_auc']:.4f})")

    # Refit on full training set
    best_pipeline.fit(X_train, y_train)

    # Test evaluation
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    auc_test = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("\n[Test Eval] (threshold=%.2f)" % threshold)
    print(f"  ROC AUC: {auc_test:.4f}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={auc_test:.3f})")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save model
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n[SAVE] Best pipeline saved to: {MODEL_PATH}")

    # SHAP explainability (only for xgb)
    shap_top5 = None
    try:
        import shap
        if "xgb" in pipelines:
            print("\n[SHAP] Running for XGBoost...")
            xgb_pipeline = pipelines["xgb"]
            if best_name != "xgb":
                xgb_pipeline.fit(X_train, y_train)

            xgb_model = xgb_pipeline.named_steps["clf"]
            pre = xgb_pipeline.named_steps["pre"]

            X_test_tx = pre.transform(X_test)
            if hasattr(X_test_tx, "toarray"):
                X_test_tx = X_test_tx.toarray()

            feat_names = get_feature_names(pre, X_test)
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_tx)

            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                sv = shap_values

            mean_abs = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
            shap_top5 = shap_df.sort_values("mean_abs_shap", ascending=False).head(5)

            print("\n[SHAP] Top 5 features:")
            print(shap_top5.to_string(index=False))
            try:
                shap.summary_plot(sv, X_test_tx, feature_names=feat_names, plot_type="bar", max_display=10, show=True)
            except Exception as e:
                print("[SHAP] summary_plot failed:", e)
        else:
            print("[SHAP] Skipped (no XGB pipeline).")
    except Exception as e:
        print("[SHAP] Skipped due to error:", e)

    return {
        "cv_results": cv_results,
        "best_name": best_name,
        "test_auc": auc_test,
        "test_metrics": {"precision": float(precision), "recall": float(recall), "f1": float(f1)},
        "shap_top5": shap_top5,
        "model_path": MODEL_PATH,
    }

if __name__ == "__main__":
    _ = train_and_eval()