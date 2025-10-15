import streamlit as st
import joblib
import re
import string
import numpy as np

# --- Configuration ---
# Set page configuration for better mobile viewing and title
st.set_page_config(page_title="Fake News Detector (SVC)", layout="centered")

# --- Model Loading and Caching (Assets still loaded, but ignored in prediction) ---
@st.cache_resource
def load_assets():
    """Loads the trained Linear SVC model and TfidfVectorizer.
       NOTE: These assets are currently NOT used by predict_news function below.
    """
    try:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        # We allow the app to run even if model files are missing, 
        # as we are using a rule-based prediction for testing.
        st.warning("Model files (.pkl) not found. Using Rule-Based Prediction for demonstration.")
        return None, None

# --- Text Preprocessing Function ---
def clean_text(text):
    """
    Cleans and preprocesses text input.
    """
    text = str(text).lower()
    text = text.replace('\n', ' ').strip() 
    text = re.sub('<.*?>+|https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(' +', ' ', text).strip()
    return text

# --- Prediction Logic (RULE-BASED DEMO) ---
def predict_news(text, model, vectorizer):
    """
    ***THIS IS A RULE-BASED FUNCTION and DOES NOT USE YOUR TRAINED MODEL.***
    It is designed to demonstrate app functionality independently of the ML model.
    """
    clean_input = clean_text(text)
    
    # RULE: If the cleaned text contains the word 'secret', it's FAKE.
    # Otherwise, it's REAL.
    if 'secret' in clean_input:
        return "FAKE"
    else:
        return "REAL"


# --- Main Streamlit App Layout ---

# Load model and vectorizer once (Note: They might be None, but that's okay for the demo)
model, vectorizer = load_assets()

st.title("📰 Fake News Credibility Detector (Demo Mode)")
st.markdown("---")
st.subheader("Rule-Based Prediction for Independence")

# Text input area for the user
article_text = st.text_area(
    "Paste the Article Text Below:", 
    height=300,
    placeholder="Enter the full text of the news article here. Try including the word 'secret'!"
)

if st.button("Detect Credibility", type="primary"):
    if not article_text or len(article_text.strip()) < 50:
        st.warning("Please enter a substantial amount of text (at least 50 characters) to analyze.")
    else:
        with st.spinner("Analyzing text and running prediction..."):
            # Run the prediction
            result = predict_news(article_text, model, vectorizer)

        st.markdown("## Prediction Result")
        st.markdown("---")
        
        # Display the result with appropriate styling
        if result == "FAKE":
            st.error("🚨 WARNING: This article is likely **FAKE NEWS** (Rule-based: contains 'secret').")
            st.markdown(
                "<p style='font-size: 18px; color: #dc3545;'>The prediction is based on a simple keyword rule, not the trained machine learning model.</p>", 
                unsafe_allow_html=True
            )
        else:
            st.success("✅ CREDIBLE: This article is likely **REAL NEWS** (Rule-based).")
            st.markdown(
                "<p style='font-size: 18px; color: #198754;'>The prediction is based on a simple keyword rule, not the trained machine learning model.</p>", 
                unsafe_allow_html=True
            )
        st.markdown("---")
