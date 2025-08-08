import streamlit as st
import joblib
import pandas as pd

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Engagement Prediction")

# Input form
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
feedback = st.selectbox("Feedback", ["Positive", "Neutral", "Negative"])

if st.button("Predict"):
    # Contoh preprocessing sederhana
    gender_val = 1 if gender == "Male" else 0
    feedback_val = {"Positive": 2, "Neutral": 1, "Negative": 0}[feedback]
    
    X = pd.DataFrame([[age, gender_val, feedback_val]], 
                     columns=["Age", "Gender", "Feedback"])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    
    st.write("Prediction:", prediction[0])
