# app.py
import streamlit as st
import pickle

# Load vectorizer & model (5000 features)
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("disaster_tweet_model.pkl", "rb"))

st.title("🧠 Disaster Tweet Classifier")

lang = st.radio("Choose Language / भाषा चुनें:", ("English", "Hindi"))
tweet_input = st.text_area("Enter the tweet here:")

if st.button("Predict 🚀"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Enter a tweet first!")
    else:
        vec = vectorizer.transform([tweet_input])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        if pred == 1:
            emoji = "🔥🚨"
            text_en = "Disaster Tweet"
            text_hi = "आपदा से संबंधित ट्वीट"
        else:
            emoji = "✅🌟"
            text_en = "Not Disaster Tweet"
            text_hi = "आपदा से संबंधित नहीं"

        st.success(f"Prediction: {text_en if lang=='English' else text_hi} {emoji}")
        st.write(f"Probability: Not Disaster={proba[0]:.2f}, Disaster={proba[1]:.2f}")
