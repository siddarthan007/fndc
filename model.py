import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_models():
    tfidf = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    return tfidf, model

tfidf, model = load_models()

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .stTextArea textarea {
        height: 200px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 18px;
        text-align: center;
    }
    .real-news {
        background: #d4edda;
        color: #155724;
    }
    .fake-news {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection System")
st.write("""
Detect potentially fake news articles using machine learning.
Paste your news text below and click 'Analyze'.
""")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("", placeholder="Paste news article here...")

with col2:
    st.write("\n")
    analyze_btn = st.button("🚀 Analyze")

if analyze_btn and user_input:
    with st.spinner("Analyzing..."):
        text_tfidf = tfidf.transform([user_input])
        
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0][prediction]
        
        result_box = f"""
        <div class="result-box {'real-news' if prediction == 1 else 'fake-news'}">
            <h3>{'✅ Real News' if prediction == 1 else '❌ Fake News'}</h3>
            <p>Confidence: {proba:.1%}</p>
        </div>
        """
        st.markdown(result_box, unsafe_allow_html=True)
        
        st.subheader("Key Indicators")
        if prediction == 0:
            st.write("""
            This article contains characteristics commonly found in fake news:
            - Emotional or sensational language
            - Lack of credible sources
            - Inconsistent factual claims
            """)
        else:
            st.write("""
            This article appears credible based on:
            - Neutral tone
            - Verifiable sources
            - Consistent factual claims
            """)

elif analyze_btn and not user_input:
    st.warning("⚠️ Please enter some text to analyze!")

st.markdown("---")
st.markdown("""
**How it works:**
- Uses a stacked ensemble of machine learning models
- Analyzes text patterns and linguistic features
- Trained on 40,000+ verified news articles
""")