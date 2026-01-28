# ============================================================
# STREAMLIT CHURN PREDICTION APP
# churn_app.py
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

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
# LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached for speed)"""
    # Update this path to your saved model
    model = joblib.load("D:\\Quantum\\Data Analyst\\Machine Learning\\saved_models\\churn_prediction_model_20260123_001139.pkl")
    
    # Load metadata
    with open("D:\\Quantum\\Data Analyst\\Machine Learning\\saved_models\\churn_prediction_model_20260123_001139_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return model, metadata

try:
    model, metadata = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Could not load model. Please check file path.")

# ============================================================
# APP TITLE AND DESCRIPTION
# ============================================================

st.title("📊 Customer Churn Prediction")
st.markdown("""
This app predicts whether a customer is likely to churn (stop using service).
Input customer details and get an instant prediction with confidence score.

**How it works:**
1. Enter customer information on the left
2. App processes the data using trained ML model
3. Model predicts: Churn or No Churn
4. Shows confidence level and recommendation
""")

# ============================================================
# SIDEBAR: INPUT FEATURES
# ============================================================

st.sidebar.header("📝 Enter Customer Details")

# Numerical features
st.sidebar.subheader("Numerical Features")

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=35,
    help="Customer age in years"
)

total_purchases = st.sidebar.number_input(
    "Total Purchases",
    min_value=1,
    max_value=25,
    value=12,
    help="Number of purchases by customer"
)

total_spend = st.sidebar.number_input(
    "Total Spend ($)",
    min_value=0.0,
    max_value=100000.0,
    value=6332.0,
    step=100.0,
    help="Total spending by customer"
)

avg_order_value = st.sidebar.number_input(
    "Average Order Value ($)",
    min_value=0.0,
    max_value=15000.0,
    value=587.0,
    step=50.0,
    help="Average value per order"
)

customer_tenure_days = st.sidebar.number_input(
    "Customer Tenure (Days)",
    min_value=1,
    max_value=3650,
    value=365,
    help="How long customer has been with us"
)

since_last_purchase = st.sidebar.number_input(
    "Days Since Last Purchase",
    min_value=1,
    max_value=2000,
    value=500,
    help="Days since last purchase (recency)"
)

recency_score = st.sidebar.slider(
    "Recency Score (0-100)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    help="Score based on purchase recency"
)

# Categorical features
st.sidebar.subheader("Categorical Features")

gender = st.sidebar.radio(
    "Gender",
    ["Male", "Female", "Other"],
    help="Customer gender"
)

city = st.sidebar.selectbox(
    "City",
    ["Bangalore", "Chennai", "Mumbai", "Hyderabad", "Delhi", "Pune"],
    help="Customer location"
)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["Beauty", "Electronics", "Fashion", "Home", "Sports"],
    help="Primary product category purchased"
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "UPI", "Net Banking"],
    help="Preferred payment method"
)

clv_segment = st.sidebar.selectbox(
    "Customer Lifetime Value Segment",
    ["Low Value", "Medium Value", "High Value", "VIP"],
    help="Customer value classification"
)

purchase_frequency = st.sidebar.number_input(
    "Purchase Frequency",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Purchases per time period"
)

# ============================================================
# PREPARE DATA FOR PREDICTION
# ============================================================

# Create feature vector matching training data
# Note: Feature order MUST match training data!

feature_dict = {
    'Age': age,
    'Total_Purchases': total_purchases,
    'Total_Spend': total_spend,
    'Avg_Order_Value': avg_order_value,
    'Customer_Tenure_Days': customer_tenure_days,
    'Since_Last_Purchase': since_last_purchase,
    'Purchase_Frequency': purchase_frequency,
    'Recency_Score': recency_score,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Other': 1 if gender == 'Other' else 0,
    'City_Chennai': 1 if city == 'Chennai' else 0,
    'City_Delhi': 1 if city == 'Delhi' else 0,
    'City_Hyderabad': 1 if city == 'Hyderabad' else 0,
    'City_Mumbai': 1 if city == 'Mumbai' else 0,
    'City_Pune': 1 if city == 'Pune' else 0,
    'Product_Category_Electronics': 1 if product_category == 'Electronics' else 0,
    'Product_Category_Fashion': 1 if product_category == 'Fashion' else 0,
    'Product_Category_Home': 1 if product_category == 'Home' else 0,
    'Product_Category_Sports': 1 if product_category == 'Sports' else 0,
    'Payment_Method_Credit Card': 1 if payment_method == 'Credit Card' else 0,
    'Payment_Method_Debit Card': 1 if payment_method == 'Debit Card' else 0,
    'Payment_Method_Net Banking': 1 if payment_method == 'Net Banking' else 0,
    'Payment_Method_UPI': 1 if payment_method == 'UPI' else 0,
    'CLV_Segment_Medium Value': 1 if clv_segment == 'Medium Value' else 0,
    'CLV_Segment_High Value': 1 if clv_segment == 'High Value' else 0,
    'CLV_Segment_VIP': 1 if clv_segment == 'VIP' else 0,
}

# ============================================================
# MAKE PREDICTION
# ============================================================

if st.sidebar.button("🔮 Predict Churn", type="primary"):
    
    if not model_loaded:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        # Create input dataframe
        input_df = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Extract probabilities
        prob_not_churn = prediction_proba[0] * 100
        prob_churn = prediction_proba[1] * 100
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        st.header("🎯 Prediction Results")
        
        # Main prediction card
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div style="background-color: #ff6b6b; padding: 20px; border-radius: 10px; color: white;">
                <h1 style="text-align: center; color: white;">⚠️ CHURN RISK</h1>
                <p style="text-align: center; font-size: 18px;">This customer is likely to churn</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #51cf66; padding: 20px; border-radius: 10px; color: white;">
                <h1 style="text-align: center; color: white;">✅ NO CHURN RISK</h1>
                <p style="text-align: center; font-size: 18px;">This customer is likely to stay</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                label="Confidence Level",
                value=f"{max(prob_churn, prob_not_churn):.1f}%",
                help="How confident is the model in this prediction?"
            )
        
        # Probability breakdown
        st.subheader("📊 Prediction Confidence")
        col1, col2 = st.columns(2)
        
        with col1:
            st.progress(prob_not_churn / 100)
            st.write(f"**No Churn:** {prob_not_churn:.2f}%")
        
        with col2:
            st.progress(prob_churn / 100)
            st.write(f"**Churn:** {prob_churn:.2f}%")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **Action Required:** This customer shows churn risk
            
            Suggested Actions:
            - 📞 Contact customer for feedback
            - 🎁 Offer targeted discount or loyalty reward
            - 📧 Send personalized retention email
            - 🎯 Recommend popular products they might like
            - 📍 Check if any service issues need resolution
            """)
        else:
            st.success("""
            **Good News:** This customer is likely to stay
            
            Suggested Actions:
            - 📈 Focus on upselling/cross-selling
            - 🏆 Enroll in VIP loyalty program
            - 📬 Regular engagement (newsletters, updates)
            - 🎉 Celebrate milestones (anniversaries, purchases)
            - 👥 Encourage referrals to friends
            """)
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            st.write(f"""
            **Model Details:**
            - Type: {metadata.get('model_type', 'N/A')}
            - Test Accuracy: {metadata.get('test_accuracy', 'N/A')}
            - Test Recall: {metadata.get('test_recall', 'N/A')} (catches % of churners)
            - Last Trained: {metadata.get('training_date', 'N/A')}
            """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
---
**Disclaimer:** This prediction is based on historical data and ML model. 
Real business decisions should combine this with human judgment and domain expertise.

**Questions?** Contact the Data Science team.
""")