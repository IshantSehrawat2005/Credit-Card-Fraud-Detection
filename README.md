# 💳 Credit Card Fraud Detection using Machine Learning

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange)

## 📌 Project Overview

This project aims to detect fraudulent credit card transactions using various Machine Learning algorithms. It leverages real-world anonymized credit card transaction data and applies data preprocessing, exploratory analysis, and classification models to predict fraudulent transactions.

---

## 📂 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraudulent transactions:** ~0.17% (highly imbalanced dataset)
- **Features:** 30 (anonymized + `Time`, `Amount`, `Class`)

---

## ⚙️ Technologies Used

- 🐍 Python 3.8+
- 📊 Pandas, NumPy
- 📈 Matplotlib, Seaborn
- 🤖 Scikit-learn
- 🧪 Imbalanced-learn (for SMOTE)

---

## 🧠 ML Algorithms Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (Optional)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 🔍 Key Steps

1. **Data Preprocessing**
   - Checked for nulls
   - Handled imbalance using SMOTE (Synthetic Minority Oversampling Technique)
   - Feature scaling using StandardScaler

2. **Exploratory Data Analysis (EDA)**
   - Visualized fraud vs. non-fraud transactions
   - Correlation heatmap
   - Distribution of amount and time

3. **Model Training & Evaluation**
   - Split data using `train_test_split`
   - Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC
   - Confusion matrix and classification report

4. **Model Comparison**
   - Compared models based on performance metrics
   - Selected the best model for deployment

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC Curve**

> 🚨 Special focus on **Recall** due to the critical nature of false negatives in fraud detection.

---

## 📁 Project Structure

