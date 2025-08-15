# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================
# Load model & feature columns
# =========================
with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# =========================
# App Header
# =========================
st.set_page_config(page_title="Customer Segmentation", layout="wide")

html_temp = """
<div style="background-color:#4B79A1;padding:20px;border-radius:10px">
    <h1 style="color:#fff;text-align:center;">Customer Segmentation Prediction</h1> 
    <h4 style="color:#fff;text-align:center;">Predicting Segments 0,1,2,3</h4> 
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.markdown("""
### Predict your customer segment based on profile
Features used:
- Age, Ever Married, Gender, Graduated, Profession
- Work Experience, Spending Score, Family Size, Var_1
""")

# =========================
# Input Form
# =========================
st.subheader("Customer Profile Input")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 0, 100, 30)
    Ever_Married = st.selectbox("Ever Married", ["Yes", "No"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Graduated = st.selectbox("Graduated", ["Yes", "No"])

with col2:
    Profession = st.selectbox("Profession", [
        "Artist", "Doctor", "Engineer", "Entertainment", "Executive",
        "Healthcare", "Homemaker", "Lawyer", "Marketing"
    ])
    Work_Experience = st.number_input("Work Experience (years)", 0, 50, 5)
    Spending_Score = st.number_input("Spending Score (0-100)", 0, 100, 50)

with col3:
    Family_Size = st.number_input("Family Size", 1, 20, 3)
    Var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

# =========================
# Encode Input with One-Hot
# =========================
def encode_input_onehot(Age, Ever_Married, Gender, Graduated, Profession,
                        Work_Experience, Spending_Score, Family_Size, Var_1):
    data = {
        "Age": Age,
        "Work_Experience": Work_Experience,
        "Spending_Score": Spending_Score,
        "Family_Size": Family_Size,
        "Ever_Married": Ever_Married,
        "Gender": Gender,
        "Graduated": Graduated,
        "Profession": Profession,
        "Var_1": Var_1
    }
    df = pd.DataFrame([data])
    
    # One-hot encode categorical columns sama seperti training
    cat_cols = ["Ever_Married", "Gender", "Graduated", "Profession", "Var_1"]
    df = pd.get_dummies(df, columns=cat_cols)
    
    # Pastikan kolom sesuai dengan model
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# =========================
# Prediction Button
# =========================
# =========================
# Prediction Button
# =========================
if st.button("Predict Segment"):
    input_df = encode_input_onehot(Age, Ever_Married, Gender, Graduated,
                                   Profession, Work_Experience, Spending_Score,
                                   Family_Size, Var_1)
    
    # Menjalankan prediksi probabilitas
    prediction_proba = model.predict_proba(input_df)
    
    # Mencari indeks dari probabilitas tertinggi
    predicted_class = np.argmax(prediction_proba, axis=1)[0]
    
    # Mengambil nilai probabilitas tertinggi
    predicted_proba_value = prediction_proba[0][predicted_class]

    st.markdown("### Prediction Result")
    
    # Menampilkan hanya satu hasil dengan probabilitas
    st.markdown(f"""
    <div style="background-color:#4B79A1;padding:20px;border-radius:10px">
        <h2 style='color:#fff;text-align:center;'>Customer Segment: {predicted_class}</h2>
        <h4 style='color:#fff;text-align:center;'>Probabilitas: {predicted_proba_value:.2%}</h4>
    </div>
    """, unsafe_allow_html=True)

    # Tambahan: Berikan penjelasan singkat tentang segmen (opsional)
    segment_descriptions = {
        0: "Pelanggan Loyal",
        1: "Pelanggan Potensial",
        2: "Pelanggan Tidak Aktif",
        3: "Pelanggan Baru"
    }
    
    st.markdown(f"**Penjelasan:** Pelanggan ini diprediksi sebagai **{segment_descriptions.get(predicted_class, 'Tidak Diketahui')}**.")

    # Show dataframe dengan probabilitas (2 desimal)
    prob_df = pd.DataFrame(prediction_proba, columns=[f"Segment {i}" for i in range(prediction_proba.shape[1])])
    st.dataframe(prob_df.style.format("{:.2f}"))

