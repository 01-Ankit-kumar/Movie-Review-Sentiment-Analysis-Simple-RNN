import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Mapping of words index back to words (for understanding)
word_index = imdb.get_word_index()

# Reverse dictionary
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('SimpleRNN_imdb.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], value=0, padding='pre', maxlen=500)
    return padded_review

# Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a review to predict the sentiment")

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):  # Corrected button
    preprocessed_input = preprocess_text(user_input)
    
    # Make prediction
    prediction = model.predict(preprocessed_input)

    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    # Display the result
    st.write(f"The sentiment of the review is {sentiment}")
    st.write(f"The prediction score is {prediction[0][0]}")