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

# Feature columns hasil one-hot encoding saat training
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# =========================
# App Header
# =========================
st.set_page_config(page_title="Customer Segmentation", layout="wide")

html_temp = """
<div style="background-color:#4B79A1;padding:20px;border-radius:10px">
    <h1 style="color:#fff;text-align:center;">Customer Se
