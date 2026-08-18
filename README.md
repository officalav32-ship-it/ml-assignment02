# Machine Learning Assignment - 2

## 1. Problem statement
Build and evaluate classification models on a public classification dataset and deploy an interactive Streamlit application that allows a user to upload test data, select a model, and view predictions and evaluation metrics.

## 2. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset is a binary classification dataset with **569 instances and 30 numeric features**, satisfying the assignment minimum of 500 instances and 12 features. The target represents malignant (0) and benign (1) classes in the packaged dataset.

The data is publicly available from the UCI Machine Learning Repository and is also distributed through scikit-learn.

A stratified **80:20 train-test split** is used with `random_state=42`.

## 3. GitHub Repository Link
**Replace this placeholder with your own repository URL after pushing the project:**

`https://github.com/<your-username>/<your-repository>`

## 4. Models used
The assignment's numbered model list contains five models, but its README comparison-table instruction says "all 6 models". To remove this ambiguity, this implementation contains the five required models plus **SVM as an additional sixth model**.

### Evaluation metrics
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- MCC = (TP×TN − FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
- AUC = Area Under the ROC Curve; it summarizes the model's ability to rank positive and negative observations across classification thresholds.

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| SVM (Additional 6th Model) | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy=0.9825, AUC=0.9954, F1=0.9861, MCC=0.9623. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| Decision Tree | Accuracy=0.9211, AUC=0.9163, F1=0.9362, MCC=0.8341. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| kNN | Accuracy=0.9561, AUC=0.9788, F1=0.9655, MCC=0.9054. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| Naive Bayes | Accuracy=0.9386, AUC=0.9878, F1=0.9517, MCC=0.8676. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| Random Forest | Accuracy=0.9474, AUC=0.9937, F1=0.9583, MCC=0.8869. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| SVM (Additional 6th Model) | Accuracy=0.9825, AUC=0.9950, F1=0.9861, MCC=0.9623. Compare these values with the other models; higher values indicate better classification performance for these metrics. |
| **Overall Winner** | **Logistic Regression** achieved the strongest overall result in this run when considering F1, AUC and MCC together. |

## 5. Streamlit application
The application includes:
1. CSV test-data upload.
2. Model-selection dropdown.
3. Prediction display.
4. Accuracy, AUC, Precision, Recall, F1 and MCC.
5. Confusion matrix.
6. Classification report.

Upload `test_data.csv` to reproduce the evaluation shown in `metrics.csv`.

## 6. How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 7. Deployment
Push all project files to GitHub, then create a Streamlit Community Cloud app using:
- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`

After deployment, replace the placeholder below with the live URL:

`https://<your-streamlit-app-url>`

## 8. Repository structure
```text
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── svm_additional_6th_model.pkl
    └── metadata.pkl
```
