import streamlit as st
import joblib
import json
import pandas as pd

st.title("Breast Cancer Classifier")

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

file = st.file_uploader("Upload CSV")

if file:
    data = pd.read_csv(file)
    st.write(data.head())

with open("metrics.json") as f:
    metrics_data = json.load(f)
