import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS)


# Load KMeans model
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
# Load model and vectorizer
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("🍽 Consumer Trust & Purchase Likelihood Predictor")

st.write("Enter a food product review to predict purchase likelihood.")

review_input = st.text_area("Enter Review Here:")

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        probability = model.predict_proba(vectorized)[0][1]
        prediction = model.predict(vectorized)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"Likely to Purchase Again (Confidence: {probability*100:.2f}%)")
        else:
            st.error(f"Unlikely to Purchase Again (Confidence: {(1-probability)*100:.2f}%)")

        st.write("Purchase Likelihood Probability:")
        if prediction == 1:
            st.progress(float(probability))
        else:
            st.progress(float(1 - probability))
        st.progress(float(probability))

        if prediction == 1:
            st.info("Business Insight: Customer shows high trust. Suitable for premium targeting.")
        else:
            st.info("Business Insight: Customer shows low trust. Risk mitigation strategies recommended.")

        
         # K-Means Segmentation
        cluster = kmeans_model.predict(vectorized)[0]

        st.subheader("Sentiment Segmentation")
        st.write(f"This review belongs to Segment {cluster}")
        st.info("Segmentation Insight: Review grouped based on textual similarity patterns.")