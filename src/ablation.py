"""
ablation.py
-----------
Goes over different regularization strengths (C) for logistic regression
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_CSV = Path("data/diabetes_preprocessed.csv")
OUT_CSV = Path("outputs/results/ablation_C_auc.csv")
FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def main() -> None:
    df = pd.read_csv(DATA_CSV)
    train = df[df["year"] == 2016]
    test  = df[df["year"] == 2017]
    X_train, y_train = train[FEATURES], train["outcome"]
    X_test,  y_test  = test[FEATURES],  test["outcome"]

    Cs = [0.01, 0.1, 1, 10, 100]
    rows = []
    for C in Cs:
        lr = LogisticRegression(C=C, max_iter=1000)
        lr.fit(X_train, y_train)
        auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        rows.append((C, auc))
        print(f"[ablation] C={C:>5}  AUC={auc:.4f}")

    pd.DataFrame(rows, columns=["C", "AUC"]).to_csv(OUT_CSV, index=False)
    print(f"[ablation] Saved → {OUT_CSV}")

if __name__ == "__main__":
    main()
