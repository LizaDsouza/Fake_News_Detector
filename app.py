import streamlit as st
import joblib
import re
import string
import numpy as np

# --- Configuration ---
# Set page configuration for better mobile viewing and title
st.set_page_config(page_title="Fake News Detector (SVC)", layout="centered")

# --- Model Loading and Caching ---
@st.cache_resource
def load_assets():
    """Loads the trained Linear SVC model and TfidfVectorizer."""
    try:
        
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error(
            "Model assets failed to load. Please ensure 'fake_news_model.pkl' "
            "and 'vectorizer.pkl' are in the same directory as this app.py file."
        )
        return None, None

# --- Text Preprocessing Function (ENHANCED for stability) ---
def clean_text(text):
    """
    Cleans and preprocesses text input to match the format used during model training.
    
    Enhanced to handle browser-added newlines and whitespace more robustly.
    """
    text = str(text).lower()
    
    # CRITICAL ADDITION: Replace newlines and strip all trailing/leading whitespace
    text = text.replace('\n', ' ').strip() 
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove numbers
    text = re.sub('\w*\d\w*', '', text)
    # Remove multiple spaces and strip again
    text = re.sub(' +', ' ', text).strip()
    return text

# --- Prediction Logic ---
def predict_news(text, model, vectorizer):
    """Takes clean text, vectorizes it, and returns the model prediction."""
    
    # 1. Clean the input text
    clean_input = clean_text(text)
    
    # 2. Vectorize the text using the loaded TfidfVectorizer
    vectorized_input = vectorizer.transform([clean_input])
    
    # 3. Make the prediction
    prediction = model.predict(vectorized_input)

    # Assumes 0 = FAKE, 1 = REAL based on common ML convention
    if prediction[0] == 0:
        return "FAKE"
    else:
        return "REAL"

# --- Main Streamlit App Layout ---

# Load model and vectorizer once
model, vectorizer = load_assets()

st.title("Fake News Credibility Detector")
st.markdown("---")
st.subheader("Linear SVC Model")

if model is None or vectorizer is None:
    st.warning("⚠️ Application cannot run. Please fix the model loading error shown above.")
else:
    # Text input area for the user
    article_text = st.text_area(
        "Paste the Article Text Below:", 
        height=300,
        placeholder="Enter the full text of the news article you wish to verify..."
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
                st.error("WARNING: This article is likely **FAKE NEWS**.")
                st.markdown(
                    "<p style='font-size: 18px; color: #dc3545;'>The model has high confidence that this text contains characteristics commonly found in disinformation.</p>", 
                    unsafe_allow_html=True
                )
            else:
                st.success("CREDIBLE: This article is likely **REAL NEWS**.")
                st.markdown(
                    "<p style='font-size: 18px; color: #198754;'>The model identifies characteristics consistent with verifiable reporting or credible sources.</p>", 
                    unsafe_allow_html=True
                )
            st.markdown("---")
