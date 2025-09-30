import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("champion_model.joblib")  # Upload this file to Colab

st.title("🧠 Parkinson's Disease Detection")
st.write("Upload patient features (CSV) to predict Parkinson's disease.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Input Data Preview:")
    st.dataframe(df.head())

    prediction = model.predict(df)
    result = ["Healthy" if p==0 else "Parkinson" for p in prediction]
    st.write("### Predictions:")
    st.write(result)
