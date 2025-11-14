import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load your trained pipeline
pipeline = load("poly_regression_pipeline.joblib")

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="",
    layout="centered"
)

st.title("Medical Insurance Charge Prediction")
st.write("Enter your information below and let the model estimate your insurance cost.")

# Main layout
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)

    with col2:
        children = st.number_input("Children", min_value=0, max_value=15, value=0)

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            ["southeast", "southwest", "northeast", "northwest"]
        )

# Predict button
if st.button("Predict Insurance Charges"):
    # Prepare DataFrame for pipeline
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "male" else 0],
        "bmi": [bmi],
        "children": [children],
        "smoker": [1 if smoker == "yes" else 0],
        "northeast": [1 if region == "northeast" else 0],
        "northwest": [1 if region == "northwest" else 0],
        "southeast": [1 if region == "southeast" else 0],
        "southwest": [1 if region == "southwest" else 0],
    })


    # Run prediction
    input = input_df[["age", "bmi", "smoker"]].copy()
    try:
        prediction = pipeline.predict(input)[0]
        st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")

    except Exception as e:
        st.error("An error occurred while predicting. Check your pipeline.")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("NDLLS - 23127289 - 23127351 - 23127354 - 23127398 - 23127538")
