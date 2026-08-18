"""
Training script for Machine Learning Assignment 2.

Dataset:
Breast Cancer Wisconsin (Diagnostic), a public UCI classification dataset.
The dataset is loaded through scikit-learn's packaged copy.
"""

## Import All necessary Libraries
import pickle
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Loading the dataset and splitting into test and train dataset
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()
X.columns = [c.replace(" ", "_") for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

## Initializing the Classification Models 
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

with open(MODEL_DIR / "metadata.pkl", "wb") as f:
    pickle.dump({
        "features": list(X.columns),
        "target_name": "target",
        "target_labels": {0: "malignant", 1: "benign"}
    }, f)

# pd.DataFrame(results).to_csv(BASE / "metrics.csv", index=False)
test = X_test.copy()
test["target"] = y_test.values
test.to_csv(BASE / "test_data.csv", index=False)
# print(pd.DataFrame(results).round(4))
