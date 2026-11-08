# Telco Customer Churn Prediction – ML Assignment 2

**Live App:** https://ml-assignment-2-telco-churn-jxnemjvupqykbyvhbxgxpv.streamlit.app
**GitHub Repo:** https://github.com/aayushibits2026/ml-assignment-2-telco-churn

---

## a. Problem Statement

A telecom company loses revenue every time a customer leaves (churns). Getting a new customer
costs more than keeping an existing one. So the business needs to know in advance which
customers are likely to churn, and target them with retention offers.

This project builds and compares five classification models that predict whether a customer
will churn (Yes/No) from their demographics, subscribed services and billing details. The
trained models are deployed in an interactive Streamlit app that evaluates them on uploaded
test data.

---

## b. Dataset Description

- **Source:** IBM Telco Customer Churn dataset, Kaggle
  (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Instances:** 7,043 customers
- **Features:** 19 input features (after dropping `customerID`)
- **Target:** `Churn` (Yes = 1, No = 0). This is a binary classification problem.
- **Class distribution:** 73.5% No Churn, 26.5% Churn. The data is imbalanced.

| Feature group | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

**Preprocessing**
- Dropped `customerID`, because it is an identifier with no predictive value.
- `TotalCharges` was stored as text, and 11 rows were blank. All 11 had `tenure = 0`, i.e.
  brand-new customers who had not been billed yet, so the blanks were filled with 0.
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled with `StandardScaler`.
- The 15 categorical features were one-hot encoded. `SeniorCitizen` was already 0/1.
- 80/20 train-test split, stratified on `Churn` to keep the same class ratio in both sets.
- Preprocessing and model are combined in one scikit-learn `Pipeline` and saved with `joblib`,
  so the app can take raw test CSV input directly.

---

## c. GitHub Repository Link

https://github.com/aayushibits2026/ml-assignment-2-telco-churn

```
ml-assignment-2-telco-churn/
├── app.py                  # Streamlit app
├── model_training.ipynb    # Data prep, training, evaluation
├── requirements.txt
├── README.md
├── test_data.csv           # 1,409-row held-out test set
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.csv
```

---

## d. Models Used

All five models were trained on the same training split and evaluated on the same 1,409-row
test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8098 | 0.8573 | 0.6721 | 0.5535 | 0.6070 | 0.4871 |
| Decision Tree (max_depth=6) | 0.8013 | 0.8418 | 0.6416 | 0.5695 | 0.6034 | 0.4729 |
| KNN (k=15) | 0.7963 | 0.8351 | 0.6276 | 0.5722 | 0.5986 | 0.4634 |
| Naive Bayes (Gaussian) | 0.7189 | 0.8306 | 0.4835 | 0.8636 | 0.6200 | 0.4690 |
| Support Vector Machine (RBF) | 0.8091 | 0.8240 | 0.6966 | 0.4973 | 0.5803 | 0.4722 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO |
| Decision Tree | TODO |
| KNN | TODO |
| Naive Bayes | TODO |
| Support Vector Machine | TODO |
| **Overall winner for this dataset** | TODO |

---

## Streamlit App Features

- Upload a test CSV (must include the `Churn` column)
- Dropdown to select any of the 5 trained models
- Displays Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix heatmap and classification report
- Download button for the sample `test_data.csv`
- Models are only loaded and used for prediction. No retraining happens in the app.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
