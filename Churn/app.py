# Gender -> 1 Female  0 Male
# Churn  -> 1 Yes     0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharge'

import streamlit as st
import joblib
import numpy as np

try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Failed to load model or scaler.")
    st.stop()

st.title("Churn Prediction App")
st.subheader("Enter Customer Information")
st.markdown("This app predicts whether a customer is likely to churn based on their age, tenure, gender, and monthly charges.")

st.divider()

# Input Fields
age = st.number_input("Enter age", min_value = 10, max_value = 100, value = 30)

tenure = st.number_input("Enter tenure", min_value = 0, max_value = 130, value = 10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value = 30, max_value = 150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict!")

st.divider()

# Prediction Logic
if predictbutton:
    try:
        gender_selected = 1 if gender == "Female" else 0
        X = [age, gender_selected, tenure, monthlycharge]
        X1 = np.array(X)
        X_array = scaler.transform([X1])
        prediction = model.predict(X_array)[0]
        
        # Confidence Score
        try:
            proba = model.predict_proba(X_array)[0][1]
            st.write(f"🧠 Churn Probability: {proba:.2f}")
        except:
            pass  # skip if model doesn't support it
        # Color-coded prediction output
        if prediction == 1:
            st.error("🚨 Customer will likely churn.")
        else:
            st.success("✅ Customer will likely stay.")
    except Exception as e:
        st.error("❌ Something went wrong during prediction. Please check inputs or model.")
else:
    st.write("Please enter the values and use the predict button")
