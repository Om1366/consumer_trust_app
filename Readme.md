# 🍽 Consumer Trust & Purchase Likelihood Prediction

## 📌 Project Overview

This project analyzes customer reviews from the Amazon Fine Food Reviews dataset to:

- Perform **Sentiment Segmentation** using K-Means clustering  
- Predict **Purchase Likelihood** using Logistic Regression  
- Classify customers into **Business Trust Segments**

The objective is to transform unstructured customer review text into meaningful business insights using Machine Learning and Natural Language Processing.

---

## 🎯 Business Problem Statement

In e-commerce platforms, customer reviews directly influence:

- Consumer trust  
- Purchase behavior  
- Product demand  
- Brand reputation  
- Revenue growth  

Manually analyzing thousands of reviews is inefficient.

This project builds an automated system that:

- Segments reviews based on textual similarity  
- Predicts whether a customer is likely to purchase again  
- Classifies customers into trust levels for strategic action  

---

## 🧠 Business & Economic Concepts Applied

### 1️⃣ Consumer Trust & Demand
Positive sentiment correlates with higher repeat purchase probability and sustained demand.

### 2️⃣ Revenue Optimization
High trust customers can be targeted with loyalty programs and premium offerings.

### 3️⃣ Risk Mitigation
Low trust predictions help identify potential churn risks.

### 4️⃣ Market Segmentation
Clustering reveals hidden patterns in customer behavior beyond rating scores.

---

## ⚙️ Technical Implementation

### 🔹 1. Data Preprocessing

- Lowercasing text  
- Removing special characters  
- Removing stopwords  

Purpose: Clean and standardize textual data before modeling.

---

### 🔹 2. TF-IDF Vectorization

- Converts text into numerical feature vectors  
- Captures importance of words across documents  
- Limited to 5000 most significant features  

Purpose: Transform raw text into machine-readable format.

---

### 🔹 3. K-Means Clustering (Unsupervised Learning)

- Groups reviews into 3 clusters  
- Based on textual similarity  
- Does not use rating labels  

Purpose: Perform sentiment segmentation and discover hidden textual patterns.

---

### 🔹 4. Logistic Regression (Supervised Learning)

- Binary classification model  
- Target variable:  
  - 1 → Rating ≥ 4 (Likely Purchase)  
  - 0 → Rating < 4 (Unlikely Purchase)  
- Outputs probability between 0 and 1  

Purpose: Predict customer purchase likelihood.

---

### 🔹 5. Model Evaluation

- Accuracy Score (~87%)  
- Confusion Matrix  
- Classification Report  

Purpose: Measure model performance on unseen data.

---

## 📊 Model Architecture
Raw Review
↓
Text Cleaning
↓
TF-IDF Vectorization
↓ ↓
Logistic Model K-Means
↓ ↓
Purchase Probability Cluster ID


Two independent intelligence branches:

- **Classification → Purchase Likelihood**
- **Segmentation → Text Pattern Clusters**

---

## 🚀 Deployment

The trained models were saved as:

- `logistic_model.pkl`
- `tfidf_vectorizer.pkl`
- `kmeans_model.pkl`

These models are loaded into a **Streamlit web application** that allows users to:

- Enter a review  
- View purchase probability (Pie + Bar visualization)  
- See business trust segment  
- View K-Means cluster assignment  

The deployment separates:

- **Training Phase (Google Colab)**
- **Inference Phase (Streamlit App)**

---

## 📂 Repository Structure
consumer_trust_app/
│
├── app.py
├── logistic_model.pkl
├── tfidf_vectorizer.pkl
├── kmeans_model.pkl
├── requirements.txt
└── README.md


---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## 📊 Dataset Information

Dataset: **Amazon Fine Food Reviews**  
Source: Kaggle  

Total records used: ~61,946 reviews  

Main features:
- Review Text  
- Rating Score  
- Product Information  

---

## 📈 Key Insights

- Majority of reviews are positive (4–5 stars)  
- TF-IDF improves performance compared to raw word counts  
- Logistic Regression achieved ~87% accuracy  
- Clustering groups reviews by textual patterns rather than sentiment alone  
- Purchase probability provides actionable business decision signals  

---

## 💼 Business Applications

This system helps businesses:

- Identify loyal customers  
- Improve customer retention strategies  
- Detect dissatisfaction trends  
- Optimize marketing campaigns  
- Enhance brand reputation management  

---

## 🏁 Conclusion

This project integrates:

- Natural Language Processing  
- Supervised Learning  
- Unsupervised Learning  
- Business Analytics  

to transform raw customer feedback into strategic insights for data-driven decision-making.