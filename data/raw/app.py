import streamlit as st
import pickle

# ✅ Load model
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Load vectorizer (ADD HERE 👇)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # ✅ Transform input
        transformed_text = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed_text)

        if prediction[0] == 0:
            st.error("❌ This is FAKE news")
        else:
            st.success("✅ This is REAL news")