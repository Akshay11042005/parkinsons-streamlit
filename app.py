import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("champion_model.joblib")
scaler = joblib.load("scaler.pkl")

# Columns the model was trained on
EXPECTED_COLUMNS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

# Mapping for alternative column names (if CSV is different)
COLUMN_MAP = {
    "Fo": "MDVP:Fo(Hz)",
    "Fhi": "MDVP:Fhi(Hz)",
    "Flo": "MDVP:Flo(Hz)",
    "Jitter_percent": "MDVP:Jitter(%)",
    "Jitter_abs": "MDVP:Jitter(Abs)",
    "RAP": "MDVP:RAP",
    "PPQ": "MDVP:PPQ",
    "DDP": "Jitter:DDP",
    "Shimmer": "MDVP:Shimmer",
    "Shimmer_dB": "MDVP:Shimmer(dB)",
    "APQ3": "Shimmer:APQ3",
    "APQ5": "Shimmer:APQ5",
    "APQ": "MDVP:APQ",
    "DDA": "Shimmer:DDA"
    # NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE already match
}

# Streamlit UI
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload patient features (CSV) to predict Parkinson's disease.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded data
        data = pd.read_csv(uploaded_file)

        # Apply column renaming if needed
        data = data.rename(columns=COLUMN_MAP)

        # Safe column selection
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        else:
            # Select only the required columns in correct order
            X = data[EXPECTED_COLUMNS]

            # Scale features
            X_scaled = scaler.transform(X)

            # Predict
            predictions = model.predict(X_scaled)

            # Show only prediction column
            result_df = pd.DataFrame({"Prediction": predictions})
            st.success("✅ Prediction completed")
            st.write(result_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
