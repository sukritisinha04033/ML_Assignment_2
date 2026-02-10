import streamlit as st
import joblib
import pandas as pd

st.title("Breast Cancer Classifier")

model_choice = st.selectbox("Select Model",
["Logistic","Decision Tree","KNN"])

file = st.file_uploader("Upload CSV")

if file:
    data = pd.read_csv(file)
    st.write(data.head())
