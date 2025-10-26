import streamlit as st 
import pandas as pd
import joblib
import numpy as np


model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("feature_names.pkl")


st.title("🧠 Stroke Prediction App")
st.markdown("Enter patient details below to predict stroke risk.")

# Input form
is_female = st.selectbox("Gender", ["Male", "Female"])
age = float(st.slider("Age", 0, 100, 30))
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
avg_glucose_level = float(st.slider("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0))
bmi = float(st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0))

work_type_Private = st.selectbox("Work Type: Private?", ["No", "Yes"])
smoking_status_smokes = st.selectbox("Smokes?", ["No", "Yes"])
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
Diabetic = st.selectbox("Diabetic?", ["No", "Yes"])

# Convert inputs to model-ready format
input_data = pd.DataFrame([{
    'is_female': 1 if is_female == "Female" else 0,
    'age' : age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    
    'work_type_Private': 1 if work_type_Private == "Yes" else 0,
    'smoking_status_smokes': 1 if smoking_status_smokes == "Yes" else 0,
    'bmi_category': {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}[bmi_category],
    'Diabetic': 1 if Diabetic == "Yes" else 0
}])







# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("⚠️ High risk of stroke detected.")
    else:
        st.success("✅ Low risk of stroke.")

    if probability is not None:
        st.write(f"Prediction confidence: {probability:.2%}")



