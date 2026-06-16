# 🇩🇪 GerRisk-XAI: Credit Risk Prediction & Explainable Analytics Dashboard

## Overview

GerRisk-XAI is an end-to-end Machine Learning project designed to predict customer credit risk and provide transparent model insights for lending decisions.

The project leverages the German Credit Dataset to classify loan applicants into low-risk and high-risk categories based on demographic, financial, and credit-related attributes. In addition to predictive modeling, the solution includes an interactive Streamlit dashboard that enables real-time risk assessment and model explainability.

This project demonstrates the complete data science workflow, from exploratory data analysis and model development to dashboard deployment and business interpretation.

---

## Business Problem

Financial institutions must evaluate the creditworthiness of loan applicants before approving loans.

Poor lending decisions can result in:

* Increased default rates
* Financial losses
* Inefficient allocation of capital
* Higher operational risk

The objective of this project is to develop a data-driven credit risk assessment system that supports transparent and consistent lending decisions.

---

## Dataset

**Dataset:** German Credit Dataset

**Source:** Kaggle

**Records:** 1,000 customers

**Features:** 20 input variables

**Target Variable:** Credit Risk (Good Credit / Bad Credit)

Example features include:

* Checking Account Balance
* Savings Balance
* Loan Amount
* Loan Duration
* Employment Length
* Credit History
* Housing Status
* Existing Credits
* Age
* Purpose of Loan

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Data quality assessment
* Target distribution analysis
* Credit history analysis
* Loan amount exploration
* Customer demographic profiling

### 2. Data Preprocessing

* Handling categorical variables
* One-Hot Encoding
* Train/Test Split
* Feature transformation using Scikit-Learn Pipeline

### 3. Model Development

Models evaluated:

* Logistic Regression
* Random Forest

Final selected model:

**Logistic Regression**

Reasons:

* Strong baseline performance
* High interpretability
* Suitable for credit risk applications
* Easy integration into explainable workflows

### 4. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Final Performance:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 78.5% |
| Precision | 62.7% |
| Recall    | 70.0% |
| F1-Score  | 66.1% |

The project prioritizes Recall for high-risk borrowers, as failing to identify a potentially risky customer can be significantly more costly than rejecting a low-risk applicant.

### 5. Explainability

To improve transparency and trust, model coefficients and feature importance analysis are incorporated into the dashboard.

The explainability module highlights:

* Top risk-increasing factors
* Top risk-reducing factors
* Most influential variables in lending decisions

---

## Interactive Dashboard

The Streamlit dashboard contains three main modules:

### Executive Overview

Provides a portfolio-level view of:

* Customer statistics
* Average loan amount
* Average customer age
* Bad credit rate
* Loan distribution
* Age distribution

### Customer Risk Assessment

Allows users to:

* Enter applicant information
* Generate real-time risk predictions
* Calculate risk scores
* Classify applicants into:

  * Low Risk
  * Medium Risk
  * High Risk

### Model Insights

Provides model transparency through:

* Risk driver analysis
* Protective factor analysis
* Feature importance visualization
* Business interpretation of model outputs

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

```text
GerRisk-XAI/

├── app/
│   └── app.py

├── data/
│   └── german_credit.csv

├── models/
│   └── credit_risk_model.pkl

├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   ├── 03_Evaluation.ipynb
│   └── 04_Explainability.ipynb

├── requirements.txt

└── README.md
```

## Future Improvements

Potential enhancements include:

* XGBoost and LightGBM implementation
* SHAP-based explainability
* Model monitoring and drift detection
* Docker containerization
* Cloud deployment
* API integration for real-world banking systems

---

## Key Skills Demonstrated

* Data Cleaning & Preparation
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning Classification
* Model Evaluation
* Explainable AI
* Streamlit Development
* Business Analytics
* End-to-End Data Science Workflow

