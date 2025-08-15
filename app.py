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
# Encode Input
# =========================
def encode_input(Age, Ever_Married, Gender, Graduated, Profession,
                 Work_Experience, Spending_Score, Family_Size, Var_1):
    data = {
        "Age": Age,
        "Ever_Married": 1 if Ever_Married=="Yes" else 0,
        "Gender": 1 if Gender=="Male" else 0,
        "Graduated": 1 if Graduated=="Yes" else 0,
        "Profession": [
            "Artist", "Doctor", "Engineer", "Entertainment", "Executive",
            "Healthcare", "Homemaker", "Lawyer", "Marketing"
        ].index(Profession),
        "Work_Experience": Work_Experience,
        "Spending_Score": Spending_Score,
        "Family_Size": Family_Size,
        "Var_1": ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"].index(Var_1)
    }
    df = pd.DataFrame([data])
    
    # Pastikan kolom sama urutannya dengan saat training
    df = df[feature_cols]
    return df

# =========================
# Prediction Button
# =========================
if st.button("Predict Segment"):
    input_df = encode_input(Age, Ever_Married, Gender, Graduated,
                            Profession, Work_Experience, Spending_Score,
                            Family_Size, Var_1)
    
    prediction_proba = model.predict_proba(input_df)
    prediction_class = np.argmax(prediction_proba, axis=1)[0]

    st.markdown("### Prediction Result")
    st.markdown(f"<h2 style='color:#ff4b4b;'>Customer Segment: {prediction_class}</h2>", unsafe_allow_html=True)

    st.markdown("### Probability for each Segment")
    prob_cols = st.column_
