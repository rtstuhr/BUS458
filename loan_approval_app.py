
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50; padding: 10px; color: white;'><b>🏦 Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

st.markdown("""
This application predicts whether a loan application will be **Approved** or **Denied**
based on applicant information and financial metrics.
""")

# Sidebar for better organization
st.sidebar.header("📋 Application Instructions")
st.sidebar.markdown("""
1. Enter applicant's financial details
2. Select employment sector and loan reason
3. Choose the preferred lender
4. Click **Predict Approval** to see results
""")

# Main input section
st.header("Enter Loan Applicant's Details")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Financial Information")

    # Monthly Gross Income (will be log-transformed)
    monthly_income = st.number_input(
        "Monthly Gross Income ($)",
        min_value=0,
        max_value=50000,
        value=5000,
        step=100,
        help="Enter the applicant's monthly gross income"
    )

    # Monthly Housing Payment (will be log-transformed)
    housing_payment = st.number_input(
        "Monthly Housing Payment ($)",
        min_value=0,
        max_value=10000,
        value=1500,
        step=50,
        help="Enter the applicant's monthly housing/rent payment"
    )

    # Requested Loan Amount (will be log-transformed)
    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=1000,
        max_value=100000,
        value=15000,
        step=500,
        help="Enter the requested loan amount"
    )

    # FICO Score
    fico_score = st.slider(
        "FICO Score",
        min_value=300,
        max_value=850,
        value=680,
        step=5,
        help="Enter the applicant's FICO credit score"
    )

with col2:
    st.subheader("👤 Personal Information")

    # Bankruptcy/Foreclosure
    bankruptcy = st.selectbox(
        "Bankruptcy or Foreclosure History?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Has the applicant ever filed for bankruptcy or foreclosure?"
    )

    # Employment Sector
    employment_sector = st.selectbox(
        "Employment Sector",
        options=[
            'consumer_discretionary',
            'consumer_staples',
            'energy',
            'financials',
            'health_care',
            'industrials',
            'information_technology',
            'materials',
            'real_estate',
            'utilities'
        ],
        index=3,  # Default to 'financials'
        help="Select the applicant's employment industry sector"
    )

    # Loan Reason
    loan_reason = st.selectbox(
        "Reason for Loan",
        options=[
            'credit_card_refinancing',
            'debt_conslidation',  # Note: keeping your typo from the data
            'home_improvement',
            'major_purchase',
            'other'
        ],
        index=1,  # Default to 'debt_conslidation'
        help="Select the primary reason for the loan"
    )

    # Lender Selection
    lender = st.selectbox(
        "Preferred Lender",
        options=['A', 'B', 'C'],
        index=0,  # Default to Lender A
        help="Select the lender (B = strictest, A = moderate, C = most lenient)"
    )

# Display calculated metrics
st.subheader("📈 Calculated Metrics")
col3, col4, col5 = st.columns(3)

# Calculate engineered features
monthly_income_log = np.log1p(monthly_income)
housing_payment_log = np.log1p(housing_payment)
loan_amount_log = np.log1p(loan_amount)

# Safe ratio calculations
if monthly_income_log > 0:
    debt_to_income = housing_payment_log / monthly_income_log
    loan_to_income = loan_amount_log / monthly_income_log
else:
    debt_to_income = 0
    loan_to_income = 0

with col3:
    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.3f}")
with col4:
    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.3f}")
with col5:
    annual_income = monthly_income * 12
    st.metric("Annual Income", f"${annual_income:,.0f}")

# Warning messages for high-risk indicators
if debt_to_income > 0.43:
    st.warning("⚠️ High Debt-to-Income Ratio (>43%) - May reduce approval chances")
if bankruptcy == 1:
    st.warning("⚠️ Bankruptcy/Foreclosure history detected - Significantly reduces approval odds")
if fico_score < 620:
    st.warning("⚠️ FICO Score below 620 - May result in denial")

# Create the input data as a DataFrame with engineered features
input_data = pd.DataFrame({
    "FICO_score": [fico_score],
    "Ever_Bankrupt_or_Foreclose": [bankruptcy],
    "Monthly_Gross_Income_log": [monthly_income_log],
    "Monthly_Housing_Payment_log": [housing_payment_log],
    "Requested_Loan_Amount_log": [loan_amount_log],
    "Debt_to_Income_Ratio": [debt_to_income],
    "Loan_to_Income_Ratio": [loan_to_income],
    "Reason": [loan_reason],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the categorical variables
input_data_encoded = pd.get_dummies(
    input_data,
    columns=['Reason', 'Employment_Sector', 'Lender'],
    drop_first=True
)

# 2. Add any "missing" columns the model expects (fill with 0)
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data
input_data_encoded = input_data_encoded[model_columns]

# Predict button with custom styling
st.markdown("---")
predict_button = st.button("🔍 Predict Loan Approval", use_container_width=True)

if predict_button:
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]
    prediction_proba = model.predict_proba(input_data_encoded)[0]

    # Get approval probability
    approval_probability = prediction_proba[1] * 100
    denial_probability = prediction_proba[0] * 100

    # Display result with custom styling
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    col6, col7 = st.columns(2)

    with col6:
        if prediction == 1:
            st.success("### ✅ APPROVED")
            st.markdown(f"**Approval Confidence:** {approval_probability:.1f}%")
        else:
            st.error("### ❌ DENIED")
            st.markdown(f"**Denial Confidence:** {denial_probability:.1f}%")

    with col7:
        st.markdown("**Probability Breakdown:**")
        st.progress(int(approval_probability))
        st.caption(f"Approval: {approval_probability:.1f}% | Denial: {denial_probability:.1f}%")
