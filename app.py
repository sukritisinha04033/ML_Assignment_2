import streamlit as st
import joblib
import json
import pandas as pd
from sklearn.metrics import classification_report

st.title("Breast Cancer Classifier")

# ------------------ LOAD METRICS ------------------
with open("metrics.json") as f:
    metrics_data = json.load(f)

# ------------------ MODEL DROPDOWN ------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ------------------ LOAD MODEL ------------------
if model_choice == "Logistic Regression":
    model = joblib.load("model/logistic.pkl")

elif model_choice == "Decision Tree":
    model = joblib.load("model/dt.pkl")

elif model_choice == "KNN":
    model = joblib.load("model/knn.pkl")

elif model_choice == "Naive Bayes":
    model = joblib.load("model/nb.pkl")

elif model_choice == "Random Forest":
    model = joblib.load("model/rf.pkl")

elif model_choice == "XGBoost":
    model = joblib.load("model/xgb.pkl")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload CSV")

if file is not None:
    data = pd.read_csv(file)
    st.subheader("Uploaded Data")
    st.write(data.head())

    try:
        predictions = model.predict(data)

        labels = ["Malignant" if p == 0 else "Benign" for p in predictions]

        result_df = pd.DataFrame({
            "Prediction Code": predictions,
            "Prediction Label": labels
        })

        st.subheader("Predictions")
        st.write(result_df)

        # ------------------ CLASSIFICATION REPORT ------------------
        # NOTE: Only works if CSV contains true target column
        if "target" in data.columns:
            y_true = data["target"]
            report = classification_report(y_true, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.subheader("Classification Report")
            st.dataframe(report_df)

        else:
            st.info("Upload CSV with 'target' column to see classification report.")

    except Exception as e:
        st.error("Prediction failed. Ensure CSV has correct features.")

# ------------------ METRICS DISPLAY ------------------
st.subheader("Model Evaluation Metrics")

if model_choice in metrics_data:
    df_metrics = pd.DataFrame(
        metrics_data[model_choice],
        index=[model_choice]
    )
    st.dataframe(df_metrics)
