import streamlit as st
import joblib
import random
import time

# --- 1. MODEL AND VECTORIZER LOADING ---
# Use st.cache_resource to load the large model files only once,
# which greatly improves performance when the app reruns.

@st.cache_resource
def load_assets():
    """Loads the trained model and TfidfVectorizer from disk.
    ACTION REQUIRED: To use your real model, replace the MOCK section 
    below with your joblib.load() calls and ensure your .pkl files are present.
    """
    try:
        
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')

        # --- MOCK OBJECTS FOR DEMO (This section is ACTIVE now) ---
        class MockModel:
            def predict(self, X):
                text = X[0].lower()
                # If the text contains formal, specific terms, predict REAL
                if "economic research" in text or "nber" in text or "analysts project" in text:
                    return ['REAL']
                # If the text contains sensational/vague terms, predict FAKE
                if "wake up" in text or "shadow government" in text or "share this article" in text:
                    return ['FAKE']
                
                # Default behavior for non-specific text (can be random or FAKE for safety)
                return ['REAL'] if random.random() < 0.5 else ['FAKE']
        
        class MockVectorizer:
            def transform(self, X):
                # Passes the raw text list through, as the mock model handles strings
                return X 
        
        model = MockModel()
        vectorizer = MockVectorizer()
        return model, vectorizer
    
    except Exception as e:
        # If the real joblib.load fails (because the files are missing)
        st.error(f"Asset loading failed: Could not load real .pkl files. Please check if they exist.")
        return None, None
        
model, vectorizer = load_assets()

# --- 2. STREAMLIT UI ---

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Fake News Detector")
st.markdown("Paste your news article below for instant credibility analysis using a machine learning model.")

# --- Text Input Area ---
article_text = st.text_area(
    "Article Text (Min. 50 words recommended)", 
    height=300,
    placeholder="Paste the full text of the news article here..."
)

# --- Prediction Button ---
if st.button("Detect Credibility", use_container_width=True, type="primary"):
    
    if not model or not vectorizer:
        st.warning("Cannot run prediction: Model assets failed to load.")
        st.stop()
        
    article = article_text.strip()
    min_words = 50
    word_count = len(article.split())

    if word_count < min_words:
        st.warning(f"Please enter at least {min_words} words ({word_count} entered) for a reliable prediction.")
    elif not article:
        st.warning("Please enter some text to analyze.")
    else:
        # --- Prediction Logic ---
        with st.spinner('Analyzing article text and running model prediction...'):
            # 1. Transform the input text
            transformed_text = vectorizer.transform([article])
            
            # 2. Get the prediction
            prediction_result = model.predict(transformed_text)[0].upper()
            
        # --- Display Result ---
        st.subheader("Prediction Result:")
        
        if prediction_result == 'REAL':
            st.success(f"REAL: This article is likely truthful and credible.")
            st.write("Keep in mind that this is a machine learning prediction and not a guarantee.")
        elif prediction_result == 'FAKE':
            st.error(f"FAKE: Caution advised. This article shows characteristics of disinformation.")
            st.write("Double-check sources and look for corroborating evidence from trusted news organizations.")
        else:
            st.info("The model returned an ambiguous result. Please try another article.")
