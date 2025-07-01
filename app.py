import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.save")

st.title("🩺 Diabetes Prediction App")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 10, 100)

# Predict button
if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled = scaler.transform(data)
    result = model.predict(scaled)[0]

    if result == 1:
        st.error("⚠️ Prediction: Diabetic")
    else:
        st.success("✅ Prediction: Not Diabetic")
