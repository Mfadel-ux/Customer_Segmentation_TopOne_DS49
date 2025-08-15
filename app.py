import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('XGBClassifier_Model.pkl', 'rb') as file:
    XGBClassifier_Model = pickle.load(file)

html_temp = """
<div style="background-color:#000;padding:15px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Customer Segmentation Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Predicting Customer Segments 0,1,2,3</h4> 
</div>
"""

desc_temp = """
### Customer Segmentation App
This app is used to predict customer segmentation based on their profile.

#### Features:
- Age, Ever Married, Gender, Graduated, Profession, Work Experience
- Spending Score, Family Size, Var_1
"""

# =========================
# Main App
# =========================
def main():
    stc.html(html_temp)
    
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

# =========================
# ML App
# =========================
def run_ml_app():
    st.subheader("Customer Segmentation Prediction")

    # Input form dalam 3 kolom
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", min_value=0, max_value=100, value=30)
        Ever_Married = st.selectbox("Ever Married", ["Yes", "No"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Graduated = st.selectbox("Graduated", ["Yes", "No"])
    
    with col2:
        Profession = st.selectbox("Profession", [
            "Artist", "Doctor", "Engineer", "Entertainment", "Executive",
            "Healthcare", "Homemaker", "Lawyer", "Marketing"
        ])
        Work_Experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
        Spending_Score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, value=50)
    
    with col3:
        Family_Size = st.number_input("Family Size", min_value=1, max_value=20, value=3)
        Var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

    if st.button("Predict Segment"):
        input_df = encode_input(Age, Ever_Married, Gender, Graduated,
                                Profession, Work_Experience, Spending_Score,
                                Family_Size, Var_1)
        
        prediction_proba = model.predict_proba(input_df)
        prediction_class = np.argmax(prediction_proba, axis=1)[0]

        st.subheader("Prediction Result")
        st.write(f"Customer Segment: **{prediction_class}**")

        st.subheader("Probability for each Segment")
        proba_df = pd.DataFrame(prediction_proba, columns=[f"Segment {i}" for i in range(prediction_proba.shape[1])])
        st.dataframe(proba_df.style.format("{:.2f}"))

# =========================
# Encode Input Function
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
    return df
