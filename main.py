import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple_rnn_imdb.h5')

# Preprocess input text
def preprocess_text(text, max_length=500, vocab_size=10000):
    words = text.lower().split()
    encoded_text = [word_index.get(word, 2) for word in words]  # OOV words = 2
    padded_text = sequence.pad_sequences([encoded_text], maxlen=max_length)
    return padded_text

# Predict sentiment
def predict_sentiment(review_text):
    processed_review = preprocess_text(review_text)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Streamlit UI
st.title("IMDB Sentiment Analysis")

user_input = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Prediction:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.4f}")
    else:
        st.warning("Please enter a review.")
