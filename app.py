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

# Jika tidak ada file feature_cols.pkl, hardcode nama kolom sesuai training
feature_cols = ["Age","Ever_Married","Gender","Graduated","Profession",
                "Work_Experience","Spending_Score","Family_Size","Var_1"]

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
