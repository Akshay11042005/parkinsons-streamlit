import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load trained model & scaler
# -----------------------
model = joblib.load("champion_model.joblib")   # trained ML model
scaler = joblib.load("scaler.pkl")             # scaler used during training

# Features the model was trained on
FEATURE_COLUMNS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Parkinson's Disease Prediction App")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded dataset
        data = pd.read_csv(uploaded_file)

        # Keep only required features (safe selection)
        missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        else:
            X = data[FEATURE_COLUMNS]

            # Scale data
            X_scaled = scaler.transform(X)

            # Make predictions
            predictions = model.predict(X_scaled)

            # Create result DataFrame
            results = pd.DataFrame(predictions, columns=["Prediction"])

            # Display only prediction column
            st.subheader("✅ Prediction Results")
            st.write(results)

            # Download option
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
