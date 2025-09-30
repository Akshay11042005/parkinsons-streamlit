import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Streamlit UI
# ===============================
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a dataset to train the model and then test predictions on new samples.")

# ===============================
# File uploader for dataset
# ===============================
dataset_file = st.file_uploader("📂 Upload Parkinson's Dataset (CSV)", type=["csv"])

if dataset_file is not None:
    try:
        # Load dataset
        df = pd.read_csv(dataset_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Drop "name" column if it exists
        if "name" in df.columns:
            df = df.drop(columns=["name"])

        # Features & target
        X = df.drop(columns=["status"])
        y = df["status"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model (Random Forest)
        model = RandomForestClassifier(random_state=42, n_estimators=200)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Model Evaluation")
        st.write(f"Accuracy on test data: **{acc:.3f}**")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save model + scaler
        joblib.dump(model, "trained_model.joblib")
        joblib.dump(scaler, "trained_scaler.pkl")

        # ===============================
        # File uploader for new samples
        # ===============================
        st.subheader("🔍 Upload New Patient Data for Prediction")
        new_file = st.file_uploader("📂 Upload CSV of New Samples", type=["csv"], key="predict")

        if new_file is not None:
            try:
                new_df = pd.read_csv(new_file)

                # Ensure all required columns are present
                required_cols = X.columns.tolist()
                missing_cols = [col for col in required_cols if col not in new_df.columns]

                # Fill missing columns with 0
                for col in missing_cols:
                    new_df[col] = 0.0

                # Arrange columns in the correct order
                new_df = new_df[required_cols]

                # Scale & predict
                new_scaled = scaler.transform(new_df)
                preds = model.predict(new_scaled)

                # Prepare prediction results
                predictions_df = pd.DataFrame({
                    "Prediction": ["Parkinson" if p == 1 else "Healthy" for p in preds]
                })

                st.subheader("Prediction Results")
                st.dataframe(predictions_df)

                # Download predictions
                csv = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="parkinson_predictions.csv",
                    mime="text/csv"
                )

                # Warning for missing columns
                if missing_cols:
                    st.warning(f"⚠️ Missing columns were filled with 0: {missing_cols}")

            except Exception as e:
                st.error(f"❌ Error processing new file: {e}")

    except Exception as e:
        st.error(f"❌ Error processing dataset: {e}")
