"""
subpopulation.py
----------------
Trains a shallow decision tree
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import log_loss

DATA_CSV = Path("data/diabetes_preprocessed.csv")
MODEL_DIR = Path("outputs/models")
REPORT_PATH = Path("outputs/results/subpop_rules.txt")
FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def main() -> None:
    df = pd.read_csv(DATA_CSV)
    test = df[df["year"] == 2017]
    X_test, y_test = test[FEATURES], test["outcome"]

    lr16 = joblib.load(MODEL_DIR / "logreg_2016.joblib")
    lr17 = joblib.load(MODEL_DIR / "logreg_2016.joblib") 

    p16 = lr16.predict_proba(X_test)[:, 1]
    p17 = lr17.predict_proba(X_test)[:, 1]

    loss16 = log_loss(y_test, p16, labels=[0,1], normalize=False) / len(y_test)
    loss17 = log_loss(y_test, p17, labels=[0,1], normalize=False) / len(y_test)
    loss_diff_label = (loss16 - loss17 > 0).astype(int)

    dt_sub = DecisionTreeClassifier(max_depth=3)
    dt_sub.fit(X_test, loss_diff_label)
    rules = export_text(dt_sub, feature_names=FEATURES)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(rules)
    print(f"[subpopulation] Rules saved → {REPORT_PATH}")

if __name__ == "__main__":
    main()
