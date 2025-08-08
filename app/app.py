import os
import joblib
import streamlit as st

import pandas as pd

# === Tentukan path absolut relatif terhadap file ini ===
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model_rf.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# === Load model dan scaler ===
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Prediksi Menggunakan Random Forest")

# Contoh input manual
feature_names = ['fitur1', 'fitur2', 'fitur3']  # ganti sesuai fitur kamu
input_data = []

for feature in feature_names:
    val = st.number_input(f"Masukkan {feature}", value=0.0)
    input_data.append(val)

if st.button("Prediksi"):
    df = pd.DataFrame([input_data], columns=feature_names)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    st.success(f"Hasil prediksi: {prediction[0]}")
