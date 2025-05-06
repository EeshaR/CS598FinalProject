"""
evaluation.py
-------------
Loads trained models and evaluates them on 2017 data
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, log_loss

DATA_CSV = Path("data/diabetes_preprocessed.csv")
MODEL_DIR = Path("outputs/models")
OUT_CSV = Path("outputs/results/metrics.csv")
PERMUTATIONS = 500

FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def compute_metrics(y, p_prev, p_curr, n_perm=500):
    auc_prev = roc_auc_score(y, p_prev)
    auc_curr = roc_auc_score(y, p_curr)
    loss_prev = log_loss(y, p_prev)
    loss_curr = log_loss(y, p_curr)
    delta_auc = auc_curr - auc_prev
    diffs = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        diffs.append(
            roc_auc_score(y_perm, p_curr) -
            roc_auc_score(y_perm, p_prev)
        )
    p_val = (np.array(diffs) >= delta_auc).mean()
    return auc_prev, auc_curr, loss_prev, loss_curr, delta_auc, p_val

def main() -> None:
    df = pd.read_csv(DATA_CSV)
    test = df[df["year"] == 2017]
    X_test, y_test = test[FEATURES], test["outcome"]

    # Load 2016 models
    lr16 = joblib.load(MODEL_DIR / "logreg_2016.joblib")
    dt16 = joblib.load(MODEL_DIR / "dtree_2016.joblib")

    # Train 2017 models fresh
    lr17 = LogisticRegression(max_iter=1000)
    dt17 = DecisionTreeClassifier(max_depth=5)
    lr17.fit(X_test, y_test)
    dt17.fit(X_test, y_test)

    rows = []
    for name, prev, curr in [
        ("LogReg", lr16, lr17),
        ("DecTree", dt16, dt17)
    ]:
        p_prev = prev.predict_proba(X_test)[:, 1]
        p_curr = curr.predict_proba(X_test)[:, 1]
        row = compute_metrics(y_test, p_prev, p_curr, PERMUTATIONS)
        rows.append((name, *row))

    metrics_df = pd.DataFrame(
        rows,
        columns=[
            "Model", "AUC_prev", "AUC_curr",
            "LogLoss_prev", "LogLoss_curr",
            "ΔAUC", "p_value"
        ]
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_CSV, index=False)
    print(f"[evaluation] Saved metrics → {OUT_CSV}")

if __name__ == "__main__":
    main()
