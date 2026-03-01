import matplotlib.pyplot as plt
import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -----------------------------
# Stopwords
# -----------------------------
stop_words = set(ENGLISH_STOP_WORDS)

# -----------------------------
# Load Models
# -----------------------------
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍽 Consumer Trust & Purchase Likelihood Predictor")

st.write("Enter a food product review to analyze purchase probability and segmentation insights.")

review_input = st.text_area("Enter Review Here:")

if st.button("Predict"):

    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # -----------------------------
        # Preprocess & Vectorize
        # -----------------------------
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])

        # -----------------------------
        # Logistic Regression Prediction
        # -----------------------------
        purchase_probability = model.predict_proba(vectorized)[0][1]
        prediction = model.predict(vectorized)[0]

        st.subheader("📊 Purchase Prediction Result")

        if prediction == 1:
            confidence = purchase_probability
            st.success("Likely to Purchase Again")
        else:
            confidence = 1 - purchase_probability
            st.error("Unlikely to Purchase Again")

        st.write(f"Model Confidence: {confidence*100:.2f}%")

        # -----------------------------
        # Purchase Probability (Always Class 1 Probability)
        # -----------------------------
        st.subheader("🛒 Purchase Probability Distribution")

        purchase = purchase_probability
        not_purchase = 1 - purchase_probability

        fig, ax = plt.subplots()

        ax.pie(
            [purchase, not_purchase],
            labels=["Purchase", "Not Purchase"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
            startangle=90
        )

        ax.axis("equal")
        st.pyplot(fig)

        # -----------------------------
        # BUSINESS TRUST SEGMENT
        # -----------------------------
        st.subheader("💼 Trust Segment (Business Classification)")

        if purchase_probability > 0.75:
            trust_segment = "High Trust Segment"
            st.success(trust_segment)
            st.info("Customer shows strong positive intent and high likelihood of repeat purchase.")
        elif purchase_probability > 0.40:
            trust_segment = "Moderate Trust Segment"
            st.warning(trust_segment)
            st.info("Customer shows moderate intent. Opportunity for engagement strategies.")
        else:
            trust_segment = "Low Trust Segment"
            st.error(trust_segment)
            st.info("Customer shows low trust. Risk mitigation and service improvement recommended.")

        # -----------------------------
        # TECHNICAL K-MEANS SEGMENT
        # -----------------------------
        cluster = kmeans_model.predict(vectorized)[0]

        segment_profiles = {
            0: "Cluster based on specific vocabulary patterns",
            1: "Cluster representing common review expressions",
            2: "Cluster representing alternate textual style patterns"
        }

        st.subheader("🔍 Text Pattern Cluster (K-Means)")
        st.write(f"Cluster ID: {cluster}")
        st.write(f"Cluster Profile: {segment_profiles.get(cluster, 'General review cluster')}")

        st.caption("K-Means clustering groups reviews based on textual similarity in TF-IDF feature space.")