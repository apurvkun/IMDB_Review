
import streamlit as st
import pickle
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import os

# Ensure NLTK data is available
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# Define paths to saved artifacts
MODEL_PATH = 'trained_models/logistic_regression_model.pkl'
VECTORIZER_PATH = 'vectorized_data/tfidf_vectorizer.pkl'

# Load the TfidfVectorizer and the trained model
@st.cache_resource
def load_artifacts():
    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return tfidf_vectorizer, model
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Make sure 'trained_models' and 'vectorized_data' directories exist and contain the .pkl files.")
        st.stop()

tfidf_vectorizer, model = load_artifacts()

# --- Text Preprocessing Functions (copied from notebook) ---
# Using RegexpTokenizer to tokenize words (alphanumeric sequences)
tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    return tokenizer.tokenize(text.lower())

# Get the list of English stop words
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def remove_symbols(tokens):
    cleaned_tokens = []
    for token in tokens:
        clean_token = re.sub(r'[^a-zA-Z0-9]', '', token)
        if clean_token: # Only keep non-empty tokens after cleaning
            cleaned_tokens.append(clean_token)
    return cleaned_tokens

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    # 1. Tokenization
    tokens = tokenize_text(text)
    # 2. Stop Word Removal
    tokens = remove_stopwords(tokens)
    # 3. Symbol Removal
    tokens = remove_symbols(tokens)
    # 4. Lemmatization
    tokens = lemmatize_tokens(tokens)
    # Join back to string for vectorization
    return ' '.join(tokens)

# --- Streamlit UI ---
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to predict its sentiment (Positive/Negative).')

user_input = st.text_area('Movie Review', height=150, help='Type your movie review here.')

if st.button('Analyze Sentiment'):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)

        # Vectorize the preprocessed text
        # TfidfVectorizer expects a list of documents, even if it's just one
        input_vector = tfidf_vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(input_vector)
        prediction_proba = model.predict_proba(input_vector)

        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        positive_proba = prediction_proba[0][1] * 100
        negative_proba = prediction_proba[0][0] * 100

        st.subheader('Prediction:')
        st.write(f"The sentiment of the review is: **{sentiment}**")
        st.progress(positive_proba / 100)
        st.write(f"Confidence: Positive {positive_proba:.2f}% | Negative {negative_proba:.2f}%")

    else:
        st.warning('Please enter a movie review to analyze.')
