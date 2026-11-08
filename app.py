import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📉 Telco Customer Churn – Model Evaluation")
st.caption("Upload test data, choose a trained model, and see how it performs.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---- Sidebar ----
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader("Upload test CSV (with Churn column)", type="csv")

with open("test_data.csv", "rb") as f:
    st.sidebar.download_button("Download sample test_data.csv", f, "test_data.csv")

if uploaded is None:
    st.info(" Upload a test CSV from the sidebar to begin.")
    st.subheader("Saved results of all models (from training)")
    st.dataframe(pd.read_csv("model/metrics.csv"), use_container_width=True)
    st.stop()

# ---- Data prep ----
df = pd.read_csv(uploaded)
if "Churn" not in df.columns:
    st.error("CSV must contain the 'Churn' target column.")
    st.stop()

df = df.drop(columns=["customerID"], errors="ignore")
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

X, y = df.drop(columns=["Churn"]), df["Churn"]

# ---- Predict (no retraining) ----
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

st.subheader(f"Results – {model_name}  ({len(df)} rows)")
c = st.columns(6)
c[0].metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
c[1].metric("AUC", f"{roc_auc_score(y, y_prob):.3f}")
c[2].metric("Precision", f"{precision_score(y, y_pred):.3f}")
c[3].metric("Recall", f"{recall_score(y, y_pred):.3f}")
c[4].metric("F1", f"{f1_score(y, y_pred):.3f}")
c[5].metric("MCC", f"{matthews_corrcoef(y, y_pred):.3f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification Report**")
    rep = classification_report(y, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)