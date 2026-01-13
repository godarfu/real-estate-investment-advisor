import streamlit as st
import joblib
import numpy as np

cls_model = joblib.load("classification_model.pkl")
reg_model = joblib.load("regression_model.pkl")

st.title("Real Estate Investment Advisor")

size = st.number_input("Property Size (SqFt)", min_value=200)
bhk = st.number_input("BHK", min_value=1)
age = st.number_input("Age of Property", min_value=0)

if st.button("Check Investment"):
    data = np.array([[size, bhk, age]])

    result = cls_model.predict(data)
    price = reg_model.predict(data)

    if result[0] == 1:
        st.success("GOOD Investment")
    else:
        st.error("NOT a Good Investment")

    st.write("Price after 5 years:", price[0])
