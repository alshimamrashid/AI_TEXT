import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load saved tools
@st.cache_resource
def load_tools():
    with open('tfidf1.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('ml_model1.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_tools()

# Preprocessing function (MUST MATCH NOTEBOOK EXACTLY)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# --- APP UI ---
st.set_page_config(page_title="AI Text Detector", page_icon="🤖")
st.title("🤖 AI vs Human Text Detector")
st.markdown("Outputs **0 (Human)** or **1 (AI)** based on text analysis.")

text_input = st.text_area("Enter your text here:", height=200)

if st.button("Predict"):
    if text_input:
        # INNOVATION: Show text stats
        st.write(f"**Original Text Length:** {len(text_input)} characters")
        
        # Clean text
        cleaned = clean_text(text_input)
        
        # INNOVATION: Show cleaned text
        with st.expander("View Cleaned Text"):
            st.write(cleaned)
        
        # Predict
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        # INNOVATION: Confidence score
        confidence = max(probability) * 100
        
        # Display result strictly as 0 or 1
        st.markdown("---")
        if prediction == 1:
         st.success(f"Result: **0** (Human Written)  |  Confidence: {confidence:.2f}%")
        else:
         st.success(f"Result: **1** (AI Generated)  |  Confidence: {confidence:.2f}%")
            
    else:
        st.warning("Please enter some text to analyze.")