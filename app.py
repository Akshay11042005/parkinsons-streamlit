import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Streamlit UI
# ===============================
st.title("🧠 Parkinson's Disease Detection")
st.write("Upload a dataset to train multiple models and see comparative performance.")

# ===============================
# File uploader for dataset
# ===============================
dataset_file = st.file_uploader("📂 Upload Parkinson's Dataset (CSV)", type=["csv"])

if dataset_file is not None:
    # Load dataset
    df = pd.read_csv(dataset_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop non-numeric identifiers
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    # Features & target
    X = df.drop(columns=["status"])
    y = df["status"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA option
    use_pca = st.checkbox("Apply PCA for Dimensionality Reduction?", value=False)
    if use_pca:
        n_components = st.slider("Select number of components", 2, X_train.shape[1], 10)
        pca = PCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        st.write(f"PCA applied: {n_components} components")

    # ===============================
    # Train multiple models
    # ===============================
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = {}
    st.subheader("📊 Model Comparison")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        st.write(f"**{name} Accuracy:** {acc:.3f}")

    # Select best-performing model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    st.success(f"✅ Best-performing model: **{best_model_name}** with accuracy {results[best_model_name]:.3f}")

    # Save best model + scaler + PCA
    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(scaler, "scaler.pkl")
    if use_pca:
        joblib.dump(pca, "pca.pkl")

    # ===============================
    # File uploader for new patient data
    # ===============================
    st.subheader("🔍 Upload New Patient Data for Prediction")
    new_file = st.file_uploader("📂 Upload CSV of New Samples", type=["csv"], key="predict")

    if new_file is not None:
        try:
            new_df = pd.read_csv(new_file)

            # Ensure feature columns match
            available = [col for col in X.columns if col in new_df.columns]
            missing = [col for col in X.columns if col not in new_df.columns]

            df_selected = new_df[available].copy()
            for col in missing:
                df_selected[col] = 0.0
            df_selected = df_selected[X.columns]

            # Scale & PCA transform
            new_scaled = scaler.transform(df_selected)
            if use_pca:
                new_scaled = pca.transform(new_scaled)

            # Predict with best model
            preds = best_model.predict(new_scaled)
            predictions_df = pd.DataFrame({
                "Prediction": ["Parkinson" if p == 1 else "Healthy" for p in preds]
            })

            st.subheader("Prediction Results")
            st.dataframe(predictions_df)

            # Download button
            csv = predictions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="parkinson_predictions.csv",
                mime="text/csv"
            )

            if missing:
                st.warning(f"⚠️ Missing columns filled with 0: {missing}")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
