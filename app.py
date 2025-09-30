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
    model = joblib.load("champion_model.joblib")  # Make sure this file exists
    scaler = joblib.load("scaler.pkl")            # Make sure this file exists
    return model, scaler

model, scaler = load_model_and_scaler()

# ===== Feature Names Used in Training =====
feature_columns = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
    'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# ===== File Upload =====
uploaded_file = st.file_uploader(
    "Upload CSV file containing patient features",
    type="csv"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Data Preview:")
        st.dataframe(df.head())

        # ===== Drop Non-feature Columns =====
        for col in ["status", "name", "Prediction", "Model"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # ===== Keep only features that exist =====
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = set(feature_columns) - set(df.columns)

        if missing_features:
            st.warning(f"The following required features are missing in the CSV and will be filled with 0: {missing_features}")

        # Fill missing features with 0
        for col in missing_features:
            df[col] = 0

        # Reorder columns to match training
        df = df[feature_columns]

        # Convert all values to float
        df = df.astype(float)

        # ===== Scale Features =====
        df_scaled = scaler.transform(df)

        # ===== Make Predictions =====
        prediction = model.predict(df_scaled)
        df['Prediction'] = ["Parkinson" if p==1 else "Healthy" for p in prediction]

        st.subheader("Prediction Results:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
