
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write(
    "This application will predict customer churn "
    "using machine learning models."
)

st.info("The prediction model will be added after training.")
