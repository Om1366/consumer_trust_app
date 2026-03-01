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

st.write("Enter a food product review to analyze purchase probability and sentiment segmentation.")

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

        # Model Confidence (classification certainty)
        if prediction == 1:
            confidence = purchase_probability
            st.success(f"Likely to Purchase Again")
        else:
            confidence = 1 - purchase_probability
            st.error(f"Unlikely to Purchase Again")

        st.write(f"Model Confidence: {confidence*100:.2f}%")

        # -----------------------------
        # Purchase Probability (Human Friendly)
        # -----------------------------
        st.subheader("🛒 Purchase Probability")

        st.progress(float(purchase_probability))
        st.write(f"Probability customer will purchase again: {purchase_probability*100:.2f}%")

        # -----------------------------
        # Business Insight
        # -----------------------------
        st.subheader("💼 Business Insight")

        if purchase_probability > 0.75:
            st.info("High trust customer. Suitable for premium targeting and loyalty programs.")
        elif purchase_probability > 0.40:
            st.info("Moderate trust level. Opportunity for engagement and personalized offers.")
        else:
            st.info("Low trust level. Risk mitigation and service improvement recommended.")

        # -----------------------------
        # K-Means Sentiment Segmentation
        # -----------------------------
        cluster = kmeans_model.predict(vectorized)[0]

        segment_profiles = {
            0: "Cluster based on specific vocabulary patterns",
            1: "Cluster representing common review expressions",
            2: "Cluster representing alternate textual style patterns"
        }

        st.subheader("🔍 Sentiment Segmentation")

        st.write(f"Segment ID: {cluster}")
        st.write(f"Segment Profile: {segment_profiles.get(cluster, 'General review cluster')}")

        st.caption("Segmentation groups reviews based on textual similarity patterns using K-Means clustering.")