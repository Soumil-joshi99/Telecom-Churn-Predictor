import streamlit as st
import joblib
import pandas as pd

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
MODEL_PATH = "models/churn_xgb_pipeline_v1.pkl"

try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- 3. HEADER ---
st.title("🔮 Telecom Churn Predictor")
st.write("Navigate through the tabs underneath to fill in the customer details.")

# --- 4. THE FORM (ORGANIZED WITH TABS) ---
tab_profile, tab_services, tab_billing = st.tabs(["👤 Profile", "📡 Services", "💳 Billing"])

# --- TAB 1: PROFILE ---
with tab_profile:
    st.header("Customer Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.divider()
    st.header("Loyalty")
    tenure = st.slider("Tenure (Months with company)", 0, 72, 12, help="How long have they been a customer?")

# --- TAB 2: SERVICES ---
with tab_services:
    st.header("Phone & Internet")
    c1, c2, c3 = st.columns(3)
    with c1: phone = st.selectbox("Phone Service", ["Yes", "No"])
    with c2: multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with c3: internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.divider()
    st.header("Add-on Services")
    st.caption("These services often increase stickiness.")
    c1, c2 = st.columns(2)
    with c1:
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with c2:
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# --- TAB 3: BILLING ---
with tab_billing:
    st.header("Contract & Payment")
    c1, c2 = st.columns(2)
    with c1: contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with c2: paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    st.divider()
    st.header("Financials")
    c1, c2 = st.columns(2)
    with c1: monthly = st.number_input("Monthly Charges ($)", 0.0, 300.0, 70.0, step=5.0)
    with c2: total = st.number_input("Total Charges ($)", 0.0, 15000.0, 100.0, step=50.0)


# --- 5. PREDICTION SECTION ---
st.divider()
st.subheader("Ready to predict?")

if st.button("Calculate Churn Risk 🚀", type="primary"):
    with st.spinner("Analyzing customer profile..."):
        # A. Create DataFrame
        input_data = pd.DataFrame({
            "gender": [gender], "SeniorCitizen": [senior], "Partner": [partner], "Dependents": [dependents],
            "tenure": [tenure], "PhoneService": [phone], "MultipleLines": [multiple], "InternetService": [internet],
            "OnlineSecurity": [security], "OnlineBackup": [backup], "DeviceProtection": [device],
            "TechSupport": [tech], "StreamingTV": [tv], "StreamingMovies": [movies], "Contract": [contract],
            "PaperlessBilling": [paperless], "PaymentMethod": [payment], "MonthlyCharges": [monthly],
            "TotalCharges": [total]
        })

        # B. FEATURE ENGINEERING (Crucial replication)
        services_list = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        input_data['Num_Services'] = (input_data[services_list] == 'Yes').sum(axis=1) + \
                                     (input_data['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)

        input_data['Avg_Price_Per_Service'] = input_data['MonthlyCharges'] / input_data['Num_Services'].replace(0, 1)

        input_data['Tenure_Group'] = input_data['tenure'].apply(lambda x: 'New' if x<12 else ('Little_Established' if x<24 else ('Established' if x<48 else 'Loyal')))

        # C. Predict
        try:
            probability = model_pipeline.predict_proba(input_data)[0][1]

            # D. Display Results visually
            st.markdown("---")
            cols = st.columns([1, 2, 1]) # Center the result
            with cols[1]:
                if probability > 0.5:
                    st.error(f"### 🚨 High Risk: {probability:.1%}")
                    st.progress(probability)
                    st.caption("Action advised: Send retention offer immediately.")
                else:
                    st.success(f"### ✅ Safe Customer: {probability:.1%}")
                    st.progress(probability)
                    st.caption("Customer is stable.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
