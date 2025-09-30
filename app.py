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

# ===== Process Uploaded CSV =====
if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Data Preview:")
        st.dataframe(df.head())

        # ===== Safe Column Selection =====

        # 1️⃣ Drop non-feature columns
        for col in ["status", "name", "Prediction", "Model"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # 2️⃣ Keep only features that exist and identify missing ones
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = set(feature_columns) - set(df.columns)

        if missing_features:
            st.warning(f"The following required features are missing and will be filled with 0: {missing_features}")

        # 3️⃣ Fill missing features with 0
        for col in missing_features:
            df[col] = 0

        # 4️⃣ Reorder columns to match training
        df = df[feature_columns]

        # 5️⃣ Convert all to float
        df = df.astype(float)

        # ===== Scale Features =====
        df_scaled = scaler.transform(df)

        # ===== Make Predictions =====
        prediction = model.predict(df_scaled)

        # ===== Display Results =====
        st.subheader("Prediction Results:")

        # Create prediction DataFrame (only prediction column)
        predictions_df = pd.DataFrame({
            "Prediction": ["Parkinson" if p == 1 else "Healthy" for p in prediction]
        })

        # Show predictions
        st.dataframe(predictions_df)

        # ===== Download Predictions as CSV =====
        csv = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="parkinson_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
