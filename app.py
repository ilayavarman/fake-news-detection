import streamlit as st
import pickle

st.title("📰 Fake News Detection System")

st.write("Enter any news text to check if it is Real or Fake.")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# User input
news = st.text_area("Enter news text:")

if st.button("Predict"):
    if len(news.strip()) < 5:
        st.warning("Please enter valid news text!")
    else:
        text_vec = vectorizer.transform([news])
        pred = model.predict(text_vec)[0]

        if pred == 1:
            st.success("✔ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
