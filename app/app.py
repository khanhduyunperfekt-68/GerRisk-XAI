import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb  
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="GerRisk-XAI",
    page_icon="🇩🇪",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "credit_risk_model.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "german_credit.csv"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


def get_feature_importance(model):

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = (
        model.named_steps["classifier"]
        .coef_[0]
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance_df["Abs_Coefficient"] = (
        importance_df["Coefficient"].abs()
    )

    return importance_df

st.title(
    "🇩🇪 GerRisk-XAI"
)

st.markdown(
    """
    ### Credit Risk Prediction & Explainable Analytics Dashboard
    
    This dashboard helps evaluate loan applicants,
    estimate default risk and identify key risk drivers.
    """
)

tab1, tab2, tab3 = st.tabs([
    " Executive Overview",
    " Customer Assessment",
    " Model Insights"
])
with tab1:

    st.header(
        "Executive Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "Average Age",
            round(df["age"].mean(), 1)
        )

    with col3:
        st.metric(
            "Average Loan",
            round(df["amount"].mean(), 0)
        )

    with col4:

        bad_rate = (
            (df["default"] == 2).mean()
            * 100
        )

        st.metric(
            "Bad Credit Rate",
            f"{bad_rate:.1f}%"
        )
        st.subheader(
        "Loan Amount Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    sns.histplot(
        df["amount"],
        bins=30,
        ax=ax
    )

    st.pyplot(fig)
    st.subheader(
        "Customer Age Distribution"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8,4)
    )

    sns.histplot(
        df["age"],
        bins=20,
        ax=ax2
    )

    st.pyplot(fig2)


with tab2:

    st.header(
        "Customer Risk Assessment"
    )

    st.markdown(
        "Enter applicant information and evaluate credit risk."
    )

    st.subheader(
        "Applicant Profile"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            75,
            35
        )

        personal_status = st.selectbox(
            "Personal Status",
            [
                "single male",
                "female",
                "divorced male",
                "married male"
            ]
        )

        job = st.selectbox(
            "Job",
            [
                "skilled employee",
                "unskilled resident",
                "mangement self-employed",
                "unemployed non-resident"
            ]
        )

    with col2:

        housing = st.selectbox(
            "Housing",
            [
                "own",
                "rent",
                "for free"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [1, 2]
        )

        foreign_worker = st.selectbox(
            "Foreign Worker",
            [
                "yes",
                "no"
            ]
        )

        telephone = st.selectbox(
            "Telephone",
            [
                "yes",
                "none"
            ]
        )

    st.subheader(
        "Financial Information"
    )

    col3, col4 = st.columns(2)

    with col3:

        checking_balance = st.selectbox(
            "Checking Balance",
            [
                "< 0 DM",
                "1 - 200 DM",
                "> 200 DM",
                "unknown"
            ]
        )

        savings_balance = st.selectbox(
            "Savings Balance",
            [
                "< 100 DM",
                "101 - 500 DM",
                "501 - 1000 DM",
                "> 1000 DM",
                "unknown"
            ]
        )

        amount = st.number_input(
            "Loan Amount",
            min_value=100,
            value=5000
        )

    with col4:

        installment_rate = st.selectbox(
            "Installment Rate",
            [1, 2, 3, 4]
        )

        existing_credits = st.selectbox(
            "Existing Credits",
            [1, 2, 3, 4]
        )


    st.subheader(
        "Loan Information"
    )

    col5, col6 = st.columns(2)

    with col5:

        months_loan_duration = st.slider(
            "Loan Duration (Months)",
            4,
            72,
            24
        )

        purpose = st.selectbox(
            "Purpose",
            [
                "radio/tv",
                "education",
                "furniture",
                "car (new)",
                "car (used)",
                "business",
                "domestic appliances",
                "repairs",
                "others",
                "retraining"
            ]
        )

        credit_history = st.selectbox(
            "Credit History",
            [
                "critical",
                "repaid",
                "delayed",
                "fully repaid",
                "fully repaid this bank"
            ]
        )

        employment_length = st.selectbox(
            "Employment Length",
            [
                "unemployed",
                "0 - 1 yrs",
                "1 - 4 yrs",
                "4 - 7 yrs",
                "> 7 yrs"
            ]
        )

    with col6:

        other_debtors = st.selectbox(
            "Other Debtors",
            [
                "none",
                "guarantor",
                "co-applicant"
            ]
        )

        residence_history = st.selectbox(
            "Residence History",
            [1, 2, 3, 4]
        )

        installment_plan = st.selectbox(
            "Installment Plan",
            [
                "none",
                "bank",
                "stores"
            ]
        )

        property_type = st.selectbox(
            "Property",
            [
                "real estate",
                "building society savings",
                "other",
                "unknown/none"
            ]
        )
if st.button(
        "Analyze Risk"
    ):

    input_data = pd.DataFrame({

            "checking_balance":
                [checking_balance],

            "months_loan_duration":
                [months_loan_duration],

            "credit_history":
                [credit_history],

            "purpose":
                [purpose],

            "amount":
                [amount],

            "savings_balance":
                [savings_balance],

            "employment_length":
                [employment_length],

            "installment_rate":
                [installment_rate],

            "personal_status":
                [personal_status],

            "other_debtors":
                [other_debtors],

            "residence_history":
                [residence_history],

            "property":
                [property_type],

            "age":
                [age],

            "installment_plan":
                [installment_plan],

            "housing":
                [housing],

            "existing_credits":
                [existing_credits],

            "job":
                [job],

            "dependents":
                [dependents],

            "telephone":
                [telephone],

            "foreign_worker":
                [foreign_worker]
        })

    probability = model.predict_proba(
            input_data
        )[0][1]

    risk_score = probability * 100
    st.divider()

    st.subheader(
            "Risk Assessment Result"
        )

    st.metric(
            "Risk Score",
            f"{risk_score:.1f}%"
        )

    if probability < 0.40:

            st.success(
                "🟢 LOW RISK APPLICANT"
            )

    elif probability < 0.70:

            st.warning(
                "🟡 MEDIUM RISK APPLICANT"
            )

    else:

            st.error(
                "🔴 HIGH RISK APPLICANT"
            )

            st.info(
            f"""
            Predicted probability of default:
            {risk_score:.1f}%

            Decision threshold:
            40%
            """
        )


with tab3:

    st.header(
        "Model Insights & Explainability"
    )

    st.markdown(
        """
        Understand the key factors influencing
        credit risk predictions.
        """
    )

    importance_df = get_feature_importance(
        model
    )
    st.subheader(
        "🔴 Top Features Increasing Credit Risk"
    )

    top_risk = (
        importance_df
        .sort_values(
            "Coefficient",
            ascending=False
        )
        .head(10)
    )

    fig1, ax1 = plt.subplots(
        figsize=(10,5)
    )

    sns.barplot(
        data=top_risk,
        x="Coefficient",
        y="Feature",
        ax=ax1
    )

    ax1.set_title(
        "Top Risk Drivers"
    )

    st.pyplot(fig1)
    st.subheader(
        "🟢 Top Features Reducing Credit Risk"
    )

    top_safe = (
        importance_df
        .sort_values(
            "Coefficient",
            ascending=True
        )
        .head(10)
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,5)
    )

    sns.barplot(
        data=top_safe,
        x="Coefficient",
        y="Feature",
        ax=ax2
    )

    ax2.set_title(
        "Top Protective Factors"
    )

    st.pyplot(fig2)
    st.subheader(
        " Most Influential Features"
    )

    top20 = (
        importance_df
        .sort_values(
            "Abs_Coefficient",
            ascending=False
        )
        .head(20)
    )

    fig3, ax3 = plt.subplots(
        figsize=(12,8)
    )

    sns.barplot(
        data=top20,
        x="Abs_Coefficient",
        y="Feature",
        ax=ax3
    )

    ax3.set_title(
        "Top 20 Most Influential Features"
    )

    st.pyplot(fig3)
    st.subheader(
        " Model Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            Algorithm

            - Logistic Regression
            - Binary Classification
            - Credit Risk Prediction
            """
        )

    with col2:

        st.info(
            """
            Model Setup

            - Features: 20
            - Training Data: 800
            - Test Data: 200
            - Decision Threshold: 0.40
            """
        )
    st.subheader(
        " Business Interpretation"
    )

    st.success(
        """
        The model identifies the most important
        factors influencing credit default risk.

        Positive coefficients indicate higher
        probability of default.

        Negative coefficients indicate lower
        probability of default.

        These insights improve transparency
        and support data-driven lending decisions.
        """
    )
