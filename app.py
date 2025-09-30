import streamlit as st
import pandas as pd
import joblib

# =======================
# Load model and scaler
# =======================
model = joblib.load("champion_model.joblib")
scaler = joblib.load("scaler.pkl")

# Required feature columns from training
required_columns = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# =======================
# Streamlit UI
# =======================
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload patient data (CSV) to predict Parkinson's disease.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        # -------------------------------
        # Safe column selection
        # -------------------------------
        # Keep only required columns, add missing ones as 0
        available = [col for col in required_columns if col in df.columns]
        missing = [col for col in required_columns if col not in df.columns]

        # Select available columns
        df_selected = df[available].copy()

        # Add missing columns filled with 0.0
        for col in missing:
            df_selected[col] = 0.0

        # Reorder to match training
        df_selected = df_selected[required_columns]

        # =======================
        # Scale + Predict
        # =======================
        df_scaled = scaler.transform(df_selected)
        prediction = model.predict(df_scaled)

        # Convert prediction to labels
        predictions_df = pd.DataFrame({
            "Prediction": ["Parkinson" if p == 1 else "Healthy" for p in prediction]
        })

        # Show results
        st.subheader("Prediction Results:")
        st.dataframe(predictions_df)

        # Download button
        csv = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="parkinson_predictions.csv",
            mime="text/csv"
        )

        # Warn if missing columns
        if missing:
            st.warning(f"⚠️ Missing columns filled with 0: {missing}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
