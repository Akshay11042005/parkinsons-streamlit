# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Page Config =====
st.set_page_config(page_title="🧠 Parkinson's Disease Detection", layout="wide")
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a CSV with patient features to predict Parkinson's disease.")

# ===== Load Pretrained Model & Scaler =====
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("champion_model.joblib")  # must be in same folder as app.py
    scaler = joblib.load("scaler.pkl")            # must be in same folder as app.py
    return model, scaler

model, scaler = load_model_and_scaler()

# ===== File Upload =====
uploaded_file = st.file_uploader(
    "Upload CSV file containing patient features",
    type="csv"
)

# ===== Process Uploaded CSV =====
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Data Preview:")
        st.dataframe(df.head())

        # Drop non-feature columns
        for col in ["status", "name", "Prediction", "Model"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure columns match training
        feature_columns = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
            'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        df = df[feature_columns]

        # Convert all values to float
        df = df.astype(float)

        # Scale features
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)
        df['Prediction'] = ["Parkinson" if p==1 else "Healthy" for p in prediction]

        st.subheader("Prediction Results:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
