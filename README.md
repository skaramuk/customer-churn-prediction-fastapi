# 🏦 Customer Churn Prediction – End-to-End ML Application

Customer churn is a critical problem in the banking sector, as retaining existing customers is often more cost-effective than acquiring new ones.  
This project aims to predict customer churn using machine learning and to estimate the **probability of churn**, enabling proactive and data-driven business decisions.

The project is implemented as an **end-to-end machine learning application**, covering the full pipeline from data analysis to model deployment with a web-based user interface.

---

## 📌 Project Overview

- **Problem Type:** Binary Classification (Churn / No Churn)
- **Target Variable:** Customer Attrition
- **Approach:** Probability-based churn prediction
- **Deployment:** FastAPI + HTML/CSS interface
- **Output:** Churn class and churn probability

---

## 🧠 Methodology

### 1️⃣ Data Analysis & Feature Engineering
- Exploratory Data Analysis (EDA)
- Handling categorical variables (One-Hot Encoding & Label Encoding)
- Feature scaling
- Awareness of class imbalance (churn rate ≈ 16%)

### 2️⃣ Model Development
Multiple classification algorithms were evaluated and compared:
- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Gradient Boosting (selected as the final model)

**Gradient Boosting** achieved the best overall performance after hyperparameter tuning using **GridSearchCV**.

### 3️⃣ Model Evaluation
- ROC-AUC score used as the primary evaluation metric
- Probability-based predictions using `predict_proba`
- Threshold-based risk interpretation:
  - 🟢 Low Risk
  - 🟡 Medium Risk
  - 🔴 High Risk

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **FastAPI**
- **HTML / CSS**
- **Uvicorn**

---

## 🚀 Application Architecture

customer-churn-prediction-fastapi/
├── app.py # FastAPI backend
├── index.html # Web-based user interface
├── churn_model.pkl # Trained machine learning model
├── requirements.txt # Project dependencies
└── README.md


👤 Author
Semih
Industrial Engineering | Data Science & Machine Learning Enthusiast
