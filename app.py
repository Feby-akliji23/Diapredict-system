import streamlit as st
import pandas as pd
from joblib import load

# Harus sama persis dengan train_model.py
NORM_COL = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
FEATURES = NORM_COL + ['Gender']


@st.cache_resource
def load_artifacts():
    return load("diabetes_model.joblib"), load("scaler.joblib")


# Load pre-trained model dan scaler
try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run `python train_model.py` first.")
    st.stop()

if list(getattr(scaler, "feature_names_in_", [])) != NORM_COL:
    st.error("Model/scaler tidak cocok dengan aplikasi ini. Jalankan ulang `python train_model.py`.")
    st.stop()

# Judul aplikasi
st.title("Diabetes Prediction Dashboard")

st.markdown("""
Masukkan hasil pemeriksaan berikut untuk memprediksi apakah seseorang terindikasi **Diabetes** atau **Non-Diabetes**.
""")

# Input untuk prediksi
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age (tahun)", min_value=1, max_value=120, value=50, step=1)
    urea = st.number_input("Urea (mmol/L)", min_value=0.0, value=4.6, step=0.1,
                           help="Kadar urea dalam darah")
    cr = st.number_input("Cr / Kreatinin (µmol/L)", min_value=0.0, value=59.0, step=1.0,
                         help="Kadar kreatinin dalam darah")
    hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.5, step=0.1,
                            help="Rata-rata gula darah 2-3 bulan terakhir. ≥6.5% umumnya indikasi diabetes")
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=24.0, step=0.1,
                          help="Body Mass Index")

with col2:
    chol = st.number_input("Chol / Kolesterol total (mmol/L)", min_value=0.0, value=4.8, step=0.1)
    tg = st.number_input("TG / Trigliserida (mmol/L)", min_value=0.0, value=1.5, step=0.1)
    hdl = st.number_input("HDL (mmol/L)", min_value=0.0, value=1.1, step=0.1,
                          help="Kolesterol baik")
    ldl = st.number_input("LDL (mmol/L)", min_value=0.0, value=2.5, step=0.1,
                          help="Kolesterol jahat")
    vldl = st.number_input("VLDL (mmol/L)", min_value=0.0, value=0.8, step=0.1,
                           help="Kolesterol jahat jenis VLDL")

# Prediksi
if st.button("Predict Diabetes"):
    raw = pd.DataFrame([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]], columns=NORM_COL)
    input_data = pd.DataFrame(scaler.transform(raw), columns=NORM_COL)
    input_data['Gender'] = 1 if gender == "Male" else 0
    input_data = input_data[FEATURES]

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Hasil prediksi: **Diabetes** (probabilitas {proba:.0%})")
    else:
        st.success(f"Hasil prediksi: **Non-Diabetes** (probabilitas diabetes {proba:.0%})")

    st.caption("Hasil ini hanya prediksi model machine learning, bukan diagnosis medis. Konsultasikan dengan dokter.")
