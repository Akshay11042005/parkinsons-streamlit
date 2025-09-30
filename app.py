import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== Streamlit UI =====
st.title("🧠 Parkinson's Disease Detection - Multi-Model with PCA")
st.write("Upload a dataset to train multiple models, apply PCA, and predict on new patient data.")

# ===== Dataset Upload =====
dataset_file = st.file_uploader("📂 Upload Parkinson's Dataset (CSV)", type=["csv"])

if dataset_file is not None:
    try:
        df = pd.read_csv(dataset_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Drop non-numeric column
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

        # Apply PCA (optional)
        n_components = st.slider("Number of PCA components", min_value=5, max_value=min(X_train.shape[1], 50), value=20)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        st.write(f"Explained variance by {n_components} components: {np.sum(pca.explained_variance_ratio_):.3f}")

        # ===== Models =====
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
        }

        results = {}
        st.subheader("📊 Model Evaluation")
        for name, model in models.items():
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            st.write(f"**{name} Accuracy:** {acc:.3f}")
            st.text(classification_report(y_test, y_pred))

        # Bar chart comparison
        st.subheader("📊 Model Accuracy Comparison")
        fig, ax = plt.subplots()
        sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")
        st.pyplot(fig)

        # Save the best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        joblib.dump(best_model, "best_model.joblib")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(pca, "pca.pkl")
        st.success(f"✅ Best model ({best_model_name}) saved successfully!")

        # ===== New Patient Prediction =====
        st.subheader("🔍 Upload New Patient Data for Prediction")
        new_file = st.file_uploader("📂 Upload CSV of New Samples", type=["csv"], key="predict")

        if new_file is not None:
            try:
                new_df = pd.read_csv(new_file)
                available = [col for col in X.columns if col in new_df.columns]
                missing = [col for col in X.columns if col not in new_df.columns]

                df_selected = new_df[available].copy()
                for col in missing:
                    df_selected[col] = 0.0
                df_selected = df_selected[X.columns]

                # Scale and PCA transform
                new_scaled = scaler.transform(df_selected)
                new_pca = pca.transform(new_scaled)

                preds = best_model.predict(new_pca)
                predictions_df = pd.DataFrame({"Prediction": ["Parkinson" if p==1 else "Healthy" for p in preds]})

                # Colored labels
                def color_label(pred):
                    return f'<span style="color:green;font-weight:bold;">✅ {pred}</span>' if pred=="Healthy" \
                        else f'<span style="color:red;font-weight:bold;">❌ {pred}</span>'
                predictions_df_colored = predictions_df.copy()
                predictions_df_colored["Prediction"] = predictions_df_colored["Prediction"].apply(color_label)

                st.subheader("Prediction Results")
                st.markdown(predictions_df_colored.to_html(escape=False, index=False), unsafe_allow_html=True)

                # Download CSV
                csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="parkinson_predictions.csv",
                    mime="text/csv"
                )

                if missing:
                    st.warning(f"⚠️ Missing columns filled with 0: {missing}")

            except Exception as e:
                st.error(f"❌ Error processing new patient file: {e}")

    except Exception as e:
        st.error(f"❌ Error processing dataset file: {e}")
