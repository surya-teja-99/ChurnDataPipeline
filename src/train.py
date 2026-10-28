"""Training: a small churn classifier, run only when validation passes.

Model choice is deliberately boring: logistic regression on a handful
of features. The point of this project is the pipeline and the gate,
not squeezing the last point of accuracy, so the model stays simple
and explainable.

Features:
  - sessions_30d, avg_session_min, support_tickets, monthly_spend
  - tenure_days (last_active_date - signup_date)
  - plan, one-hot encoded

The trained model is pickled to models/ alongside a metrics JSON.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import config

FEATURES = ["sessions_30d", "avg_session_min", "support_tickets",
            "monthly_spend", "tenure_days"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data["tenure_days"] = (
        pd.to_datetime(data["last_active_date"])
        - pd.to_datetime(data["signup_date"])
    ).dt.days.clip(lower=0)
    X = pd.get_dummies(data[FEATURES + ["plan"]], columns=["plan"])
    y = data["churned"].astype(int)
    return X, y


def train(df: pd.DataFrame, date_str: str) -> dict:
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "date": date_str,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "features": list(X.columns),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / f"churn_model_{date_str}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metrics["model_path"] = str(model_path)
    return metrics
