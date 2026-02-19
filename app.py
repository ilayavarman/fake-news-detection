import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("📰 Fake News Detection System (Train + Predict)")

st.write("This app trains a Machine Learning model and predicts Fake or Real News.")

# ------------------------------
# TRAINING SECTION
# ------------------------------
st.header("🔧 Training Model")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✔ Model Trained Successfully! Accuracy: {acc*100:.2f}%")

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

st.info("Model and vectorizer saved as model.pkl and vectorizer.pkl")

# ------------------------------
# PREDICTION SECTION
# ------------------------------
st.header("📝 Predict News")

user_input = st.text_area("Enter News Text:")

if st.button("Predict Fake/Real"):
    if user_input.strip() == "":
        st.warning("Please enter text to predict.")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]

        if prediction == 1:
            st.success("✔ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
