import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Fraud Detection System (Email + Financial)")

option = st.selectbox(
    "Choose Detection Type",
    ("Email Fraud Detection", "Financial Fraud Detection")
)

# ================= EMAIL FRAUD =================
if option == "Email Fraud Detection":

    df = pd.read_csv("phishing_small.csv")

    X = df['text_combined']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    nb = MultinomialNB()
    nb.fit(X_vec, y)

    svm = LinearSVC()
    svm.fit(X_vec, y)

    user_input = st.text_area("Enter Email Content")

    if st.button("Check Email Fraud"):
        input_vec = vectorizer.transform([user_input])

        nb_pred = nb.predict(input_vec)[0]
        svm_pred = svm.predict(input_vec)[0]

        hybrid_pred = 1 if nb_pred == 1 or svm_pred == 1 else 0

        if hybrid_pred == 1:
            st.error("🚨 Fraudulent Email")
        else:
            st.success("✅ Safe Email")


# ================= FINANCIAL FRAUD =================
elif option == "Financial Fraud Detection":

    df = pd.read_csv("fraud_small.csv")
    df.columns = df.columns.str.strip()

    df = df.sample(5000, random_state=42)   # reduced for speed

    # Encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Target column
    if 'isFraud' in df.columns:
        target_col = 'isFraud'
    elif 'Class' in df.columns:
        target_col = 'Class'
    else:
        st.error("Target column not found")
        st.stop()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nb = GaussianNB()
    nb.fit(X_scaled, y)

    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=100)
    mlp.fit(X_scaled, y)

    st.write("Enter Transaction Details:")

    # Take only first 5 features for input (simplified UI)
    input_data = []
    for col in X.columns[:5]:
        val = st.number_input(f"{col}")
        input_data.append(val)

    if st.button("Check Financial Fraud"):

        # fill remaining features with 0
        remaining = [0]*(X.shape[1] - 5)
        final_input = np.array(input_data + remaining).reshape(1, -1)

        final_input_scaled = scaler.transform(final_input)

        nb_pred = nb.predict(final_input_scaled)[0]
        mlp_pred = mlp.predict(final_input_scaled)[0]

        hybrid_pred = 1 if nb_pred == 1 or mlp_pred == 1 else 0

        if hybrid_pred == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legit Transaction")
