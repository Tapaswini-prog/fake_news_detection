import pickle

# Load model
with open("models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_news(text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)

    return "FAKE ❌" if prediction[0] == 0 else "REAL ✅"