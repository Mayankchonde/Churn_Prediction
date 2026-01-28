# ============================================================
# STREAMLIT CHURN PREDICTION APP
# app2.py
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FILE PATHS (STREAMLIT CLOUD SAFE)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "churn_prediction_model_20260128_140235.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR, "churn_prediction_model_20260128_140235_metadata.json"
)

# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        return model, metadata

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None


model, metadata = load_model()
model_loaded = model is not None

# ============================================================
# APP TITLE AND DESCRIPTION
# ============================================================

st.title("📊 Customer Churn Prediction")

st.markdown("""
This app predicts whether a customer is likely to churn (stop using the service).

**How it works:**
1. Enter customer details from the sidebar  
2. Model processes the data  
3. Predicts churn risk with confidence  
4. Shows business recommendations  
""")

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("📝 Enter Customer Details")

# ---------------- Numerical Features ----------------

st.sidebar.subheader("Numerical Features")

age = st.sidebar.slider("Age", 18, 80, 35)
total_purchases = st.sidebar.number_input("Total Purchases", 1, 25, 12)
total_spend = st.sidebar.number_input("Total Spend ($)", 0.0, 100000.0, 6332.0, step=100.0)
avg_order_value = st.sidebar.number_input("Average Order Value ($)", 0.0, 15000.0, 587.0, step=50.0)
customer_tenure_days = st.sidebar.number_input("Customer Tenure (Days)", 1, 3650, 365)
since_last_purchase = st.sidebar.number_input("Days Since Last Purchase", 1, 2000, 500)
purchase_frequency = st.sidebar.number_input("Purchase Frequency", 0.0, 10.0, 1.0, step=0.1)
recency_score = st.sidebar.slider("Recency Score", 0.0, 100.0, 50.0)

# ---------------- Categorical Features ----------------

st.sidebar.subheader("Categorical Features")

gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
city = st.sidebar.selectbox("City", ["Bangalore", "Chennai", "Mumbai", "Hyderabad", "Delhi", "Pune"])
product_category = st.sidebar.selectbox("Product Category", ["Beauty", "Electronics", "Fashion", "Home", "Sports"])
payment_method = st.sidebar.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking"])
clv_segment = st.sidebar.selectbox("CLV Segment", ["Low Value", "Medium Value", "High Value", "VIP"])

# ============================================================
# FEATURE VECTOR (MUST MATCH TRAINING ORDER)
# ============================================================

feature_dict = {
    'Age': age,
    'Total_Purchases': total_purchases,
    'Total_Spend': total_spend,
    'Avg_Order_Value': avg_order_value,
    'Customer_Tenure_Days': customer_tenure_days,
    'Since_Last_Purchase': since_last_purchase,
    'Purchase_Frequency': purchase_frequency,
    'Recency_Score': recency_score,

    'Gender_Male': 1 if gender == "Male" else 0,
    'Gender_Other': 1 if gender == "Other" else 0,

    'City_Chennai': 1 if city == "Chennai" else 0,
    'City_Delhi': 1 if city == "Delhi" else 0,
    'City_Hyderabad': 1 if city == "Hyderabad" else 0,
    'City_Mumbai': 1 if city == "Mumbai" else 0,
    'City_Pune': 1 if city == "Pune" else 0,

    'Product_Category_Electronics': 1 if product_category == "Electronics" else 0,
    'Product_Category_Fashion': 1 if product_category == "Fashion" else 0,
    'Product_Category_Home': 1 if product_category == "Home" else 0,
    'Product_Category_Sports': 1 if product_category == "Sports" else 0,

    'Payment_Method_Credit Card': 1 if payment_method == "Credit Card" else 0,
    'Payment_Method_Debit Card': 1 if payment_method == "Debit Card" else 0,
    'Payment_Method_Net Banking': 1 if payment_method == "Net Banking" else 0,
    'Payment_Method_UPI': 1 if payment_method == "UPI" else 0,

    'CLV_Segment_Medium Value': 1 if clv_segment == "Medium Value" else 0,
    'CLV_Segment_High Value': 1 if clv_segment == "High Value" else 0,
    'CLV_Segment_VIP': 1 if clv_segment == "VIP" else 0,
}

# ============================================================
# PREDICTION
# ============================================================

if st.sidebar.button("🔮 Predict Churn", type="primary"):

    if not model_loaded:
        st.error("❌ Model not loaded. Check deployment files.")
    else:
        input_df = pd.DataFrame([feature_dict])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        prob_no = prob[0] * 100
        prob_yes = prob[1] * 100

        st.header("🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ Customer is likely to CHURN")
            else:
                st.success("✅ Customer is likely to STAY")

        with col2:
            st.metric("Confidence", f"{max(prob_no, prob_yes):.2f}%")

        st.subheader("📊 Probability Breakdown")
        st.progress(prob_yes / 100)
        st.write(f"Churn: **{prob_yes:.2f}%**")
        st.write(f"No Churn: **{prob_no:.2f}%**")

        st.subheader("💡 Recommendation")
        if prediction == 1:
            st.warning("Target this customer with retention strategies.")
        else:
            st.success("Focus on loyalty and upselling opportunities.")

        with st.expander("ℹ️ Model Info"):
            st.write(f"Model Type: {metadata.get('model_type')}")
            st.write(f"Accuracy: {metadata.get('test_accuracy')}")
            st.write(f"Recall: {metadata.get('test_recall')}")
            st.write(f"Trained On: {metadata.get('training_date')}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("⚠️ Predictions are probabilistic. Use business judgment before action.")

