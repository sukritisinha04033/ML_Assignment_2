# Machine Learning Assignment 2 – Breast Cancer Classification

## 1. Problem Statement
The objective of this project is to build and compare multiple Machine Learning classification models to predict whether a tumor is **Malignant** or **Benign** using medical diagnostic features.  
The project also includes developing and deploying an interactive **Streamlit Web Application** that allows users to upload test data, select a model, and view predictions along with evaluation metrics.

---

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Access Method:** Loaded using Scikit-Learn built-in dataset  

### Dataset Details
- **Number of Instances:** 569
- **Number of Features:** 30 numerical features
- **Target Classes:**
  - 0 → Malignant
  - 1 → Benign
- **Type:** Binary Classification
- **Feature Examples:**
  - Mean Radius
  - Mean Texture
  - Mean Area
  - Worst Perimeter
  - Worst Concavity
  - Worst Symmetry

The dataset satisfies assignment requirements:
- Minimum 12 features ✔
- Minimum 500 instances ✔

---

## 3. Machine Learning Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## 4. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.97 | 0.99 | 0.97 | 0.98 | 0.97 | 0.94 |
| Decision Tree | 0.94 | 0.94 | 0.93 | 0.95 | 0.94 | 0.88 |
| KNN | 0.95 | 0.97 | 0.96 | 0.95 | 0.95 | 0.90 |
| Naive Bayes | 0.93 | 0.96 | 0.92 | 0.94 | 0.93 | 0.86 |
| Random Forest | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 |
| XGBoost | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 |

*(Values may vary slightly depending on random state.)*

---

## 6. Observations on Model Performance

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Performed very well with high precision and recall. Suitable baseline model. |
| Decision Tree | Easy to interpret but slightly lower accuracy compared to ensemble models. |
| KNN | Good performance but sensitive to feature scaling and data distribution. |
| Naive Bayes | Fast and simple but assumes feature independence, which may reduce accuracy. |
| Random Forest | Excellent performance due to ensemble averaging and reduced overfitting. |
| XGBoost | Best overall performance with highest AUC and MCC scores. Very robust model. |

---

## 7. Streamlit Web Application Features

The deployed Streamlit app provides:

- CSV Dataset Upload  
- Model Selection Dropdown  
- Prediction Output (Malignant / Benign)  
- Evaluation Metrics Display  
- Classification Report  
- Interactive User Interface  

---

## 8. Project Structure

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- metrics.json
│-- model/
├ logistic.pkl
├ dt.pkl
├ knn.pkl
├ nb.pkl
├ rf.pkl
└ xgb.pkl


---

## 9. Requirements

streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost
joblib

## 10. Deployment

The application is deployed using **Streamlit Community Cloud**.  
Users can access the live app via the provided Streamlit link.

---

## 11. Conclusion

This project demonstrates an end-to-end Machine Learning workflow including:

- Data preprocessing  
- Model training  
- Evaluation and comparison  
- Model serialization  
- Interactive web app development  
- Cloud deployment  

The ensemble models (Random Forest and XGBoost) achieved the best overall performance, while Logistic Regression provided a strong and interpretable baseline.
