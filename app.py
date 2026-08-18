import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "metadata.pkl", "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="ML Classification Model Comparison", layout="wide")
st.title("Machine Learning Classification Dashboard")
st.caption("Breast Cancer Wisconsin (Diagnostic) — model comparison")

meta = load_metadata()
features = meta["features"]

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
selected = st.sidebar.selectbox("Select classification model", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload the supplied test_data.csv from the repository to evaluate a model.")
    st.write("Required feature columns:", ", ".join(features))
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X = df[features]
has_target = "target" in df.columns

model = load_model(MODEL_DIR / MODEL_FILES[selected])
pred = model.predict(X)

st.subheader(f"Predictions — {selected}")
result = df.copy()
result["predicted_target"] = pred
result["predicted_class"] = np.where(pred == 1, "benign", "malignant")
st.dataframe(result, use_container_width=True)

if has_target:
    y = df["target"]
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1 Score": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

    st.subheader("Evaluation Metrics")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred)
    st.dataframe(pd.DataFrame(
        cm,
        index=["Actual malignant", "Actual benign"],
        columns=["Predicted malignant", "Predicted benign"]
    ))

    st.subheader("Classification Report")
    report = classification_report(
        y, pred, target_names=["malignant", "benign"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4))
else:
    st.warning("The uploaded CSV has no 'target' column, so evaluation metrics cannot be calculated.")
