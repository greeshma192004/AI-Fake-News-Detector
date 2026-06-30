import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

st.title("AI Fake News Detector")

news = st.text_area("Paste News Article Here")

if st.button("Check News"):

    news = clean_text(news)

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    prob = model.predict_proba(news_vector)

    fake_prob = prob[0][0] * 100
    real_prob = prob[0][1] * 100

    st.write(f"Fake Probability: {fake_prob:.2f}%")
    st.write(f"Real Probability: {real_prob:.2f}%")

    if prediction[0] == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")
