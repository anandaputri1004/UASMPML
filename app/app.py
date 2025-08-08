import os
import joblib
import streamlit as st

import pandas as pd

# Path ke model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_package.pkl")

# Load model
try:
    model_pkg = joblib.load(MODEL_PATH)
    model = model_pkg['model']
    scaler = model_pkg['scaler']
    feature_names = model_pkg['feature_names']
    
    st.title("Prediksi Pembelian Makanan Online")
    
    # Input form
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.number_input(f"Masukkan {feature}", value=0.0)
    
    if st.button("Prediksi"):
        input_df = pd.DataFrame([inputs], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Hasil: {'Beli' if prediction[0] == 1 else 'Tidak Beli'}")
        
except FileNotFoundError:
    st.error("Error: Model tidak ditemukan. Pastikan file model_package.pkl ada di direktori yang benar.")
except Exception as e:
    st.error(f"Terjadi error: {str(e)}")