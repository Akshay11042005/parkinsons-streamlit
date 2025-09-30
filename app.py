# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page config
st.set_page_config(page_title="Parkinson's Disease Classification", layout="wide")

# ====== App Title ======
st.title("Parkinson's Disease Classification")
st.write("Upload your Parkinson's dataset and select models to predict.")

# ====== Upload Dataset ======
uploaded_file = st.file_uploader("Upload Parkinson's CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Drop non-numeric identifiers
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    # Prepare features and target
    X = df.drop(columns=["status"])
    y = df["status"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ====== Sidebar Options ======
    st.sidebar.header("Options")
    use_pca = st.sidebar.checkbox("Use PCA (10 components)")
    model_choice = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "SVM", "MLP Neural Network"],
        default=["Random Forest"]
    )

    # Apply PCA if selected
    if use_pca:
        pca = PCA(n_components=10, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        st.write(f"Explained variance by 10 components: {np.sum(pca.explained_variance_ratio_):.3f}")

    # Train-test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ====== Helper Function ======
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.3f}")
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Healthy', 'Parkinson'],
                    yticklabels=['Healthy', 'Parkinson'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        return acc

    # ====== Model Initialization ======
    results = {}
    if "Random Forest" in model_choice:
        rf = RandomForestClassifier(random_state=42, n_estimators=200)
        st.subheader("Random Forest Results")
        results["Random Forest"] = evaluate_model(rf, X_train, X_test, y_train, y_test)

    if "SVM" in model_choice:
        svm = SVC(kernel="rbf", random_state=42)
        st.subheader("SVM Results")
        results["SVM"] = evaluate_model(svm, X_train, X_test, y_train, y_test)

    if "MLP Neural Network" in model_choice:
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        st.subheader("MLP Neural Network Results")
        results["MLP Neural Network"] = evaluate_model(mlp, X_train, X_test, y_train, y_test)

    # ====== Results Summary ======
    st.subheader("Accuracy Summary")
    summary_df = pd.DataFrame({"Model": list(results.keys()), "Accuracy": list(results.values())})
    st.dataframe(summary_df)

    # ====== Feature Importance (Random Forest only) ======
    if "Random Forest" in model_choice:
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.subheader("Top 10 Important Features")
        st.dataframe(feature_importance.head(10))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        top10 = feature_importance.head(10)
        ax2.barh(top10["Feature"], top10["Importance"], color=plt.cm.viridis(np.linspace(0, 1, 10)))
        ax2.invert_yaxis()
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("Top 10 Features for Parkinson's Detection")
        st.pyplot(fig2)
