# src/models/baseline.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# You'll need to install this library: pip install xgboost imbalanced-learn
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Install with: pip install xgboost")
    XGBClassifier = None


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("imbalanced-learn not found. Install with: pip install imbalanced-learn")
    SMOTE = None
    ImbPipeline = None


def build_preprocessor(df: pd.DataFrame, target: str = "stroke") -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create a ColumnTransformer that scales numeric cols and one-hot encodes categoricals.
    Works whether your data is raw (categoricals present) or processed (mostly numeric)."""
    drop_cols = [c for c in ["id", target] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if len(cat_cols) else
            ("cat", "drop", [])
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre, num_cols, cat_cols


def build_pipelines(
    preprocessor: ColumnTransformer,
    y_train: pd.Series | np.ndarray,
    random_state: int = 42
) -> Dict[str, Pipeline | ImbPipeline]:
    """Builds a dictionary of machine learning pipelines with imbalanced data handling."""

    # Calculate the positive-to-negative class ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / pos_count

    # Define the models
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
        random_state=random_state
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )

    models: Dict[str, Pipeline] = {
        "logreg": Pipeline([("pre", preprocessor), ("clf", lr)]),
        "rf": Pipeline([("pre", preprocessor), ("clf", rf)]),
    }

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            # This is the key change for XGBoost
            scale_pos_weight=scale_pos_weight_value,
        )
        models["xgb"] = Pipeline([("pre", preprocessor), ("clf", xgb)])

    return models