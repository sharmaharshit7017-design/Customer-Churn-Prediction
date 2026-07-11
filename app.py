import pandas as pd

import streamlit as st
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using a trained Random Forest model.")
monthly_charges =st.number_input('enter monthly charges',
                                 min_value = 0.0,
                                 value = 50.0)
st.write("Monthly Charges: ", monthly_charges)
gender = st.selectbox(
    'Select Gender',
    ['Male', 'Female']
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)



tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=72,
    value=12
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


st.divider()


model = joblib.load("best_random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
user_data = pd.DataFrame({
       
        'Contract': [contract],
        'Internet Service': [internet],
        'Monthly Charges': [monthly_charges],
        'Tenure Months': [tenure],
        'Online Security': [online_security],
        'Tech Support': [tech_support],
        'Paperless Billing': [paperless_billing],
        'Payment Method': [payment_method]
    })
predict = st.button("Predict")

user_data = pd.get_dummies(user_data, drop_first=True)
user_data = user_data.reindex(columns=feature_columns, fill_value=0)
prediction = model.predict(user_data)
probablity = model.predict_proba(user_data)
if prediction[0] == "Yes":
    st.error("❌ Customer is likely to Churn")
else:
    st.success("✅ Customer is NOT likely to Churn")
st.write(f'confidence: {max(probablity[0]) * 100:.2f}%')