# src/models/evaluate.py

from __future__ import annotations
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stroke_data_processed.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
TARGET = "stroke"
RANDOM_STATE = 42

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate(threshold: float = 0.5):
    # Load processed data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[c for c in ("id", TARGET) if c in df.columns], errors="ignore")
    y = df[TARGET].astype(int)

    # Train/test split (must match trainer.py)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[DATA] Test: {X_test.shape}")

    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"[LOAD] Model loaded from {MODEL_PATH}")

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("\n[Evaluation Results]")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # ----------------------------
    # Feature Importance (General)
    # ----------------------------
    print("\n[Feature Importance] Computing...")
    
    # Get the final estimator and preprocessor from the pipeline
    clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    
    # Get feature names after preprocessing
    try:
        feat_names = pre.get_feature_names_out(X_test.columns)
    except Exception:
        feat_names = [f"f{i}" for i in range(pre.transform(X_test).shape[1])]

    # Use SHAP for tree-based models and Permutation Importance for others
    if hasattr(clf, "feature_importances_"):
        print("Using built-in feature importance (e.g., from tree-based model)...")
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
    elif "xgb" in model.named_steps: # Use SHAP if available
        print("Using SHAP for explainability...")
        try:
            import shap
            explainer = shap.TreeExplainer(clf)
            # Transform data and get SHAP values
            X_test_tx = pre.transform(X_test)
            if hasattr(X_test_tx, "toarray"):
                X_test_tx = X_test_tx.toarray()
            
            shap_values = explainer.shap_values(X_test_tx)
            
            # For multi-class, shap_values is a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # For the positive class
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": mean_abs_shap
            }).sort_values("importance", ascending=False)
            
            # You can also generate the summary plot here
            shap.summary_plot(shap_values, X_test_tx, feature_names=feat_names, plot_type="bar")
            
        except ImportError:
            print("SHAP not found, falling back to Permutation Importance.")
            imp_df = run_permutation_importance(clf, pre, X_test, y_test, feat_names)
    else:
        print("Falling back to Permutation Importance...")
        imp_df = run_permutation_importance(clf, pre, X_test, y_test, feat_names)

    # Print and plot top features
    print(imp_df.head(10).to_string(index=False))
    plot_feature_importances(imp_df.head(10))

# ----------------------------
# Helper Functions
# ----------------------------
def run_permutation_importance(clf, preprocessor, X_test, y_test, feat_names):
    """Computes and returns permutation importance results."""
    X_test_tx = preprocessor.transform(X_test)
    if hasattr(X_test_tx, "toarray"):
        X_test_tx = X_test_tx.toarray()

    result = permutation_importance(
        clf, X_test_tx, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance", ascending=False)
    
    return imp_df

def plot_feature_importances(imp_df: pd.DataFrame):
    """Plots a bar chart of feature importances."""
    plt.figure(figsize=(8,5))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top-10 Feature Importances")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    evaluate(threshold=0.5)