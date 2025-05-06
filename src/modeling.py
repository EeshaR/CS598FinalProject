"""
modeling.py
-----------
Trains Logistic Regression and Decision Tree classifiers 
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

DATA_CSV = Path("data/diabetes_preprocessed.csv")
MODEL_DIR = Path("outputs/models")
FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

def main() -> None:
    df = pd.read_csv(DATA_CSV)
    train = df[df["year"] == 2016]
    X_train, y_train = train[FEATURES], train["outcome"]

    # Logistic Regression
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    lr.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, MODEL_DIR / "logreg_2016.joblib")
    joblib.dump(dt, MODEL_DIR / "dtree_2016.joblib")
    print("[modeling] Saved LR and DT models to outputs/models/")

if __name__ == "__main__":
    main()
