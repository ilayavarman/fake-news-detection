import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# 1. Load Dataset

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Label the data
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)


# 2. Graph: Fake % vs True %

fake_count = len(fake)
true_count = len(true)
total = fake_count + true_count

fake_percent = (fake_count / total) * 100
true_percent = (true_count / total) * 100

print(f"\nFake News: {fake_percent:.2f}%")
print(f"True News: {true_percent:.2f}%")

plt.figure(figsize=(6,6))
plt.pie([fake_percent, true_percent], labels=['Fake News','True News'], autopct='%1.2f%%')
plt.title("Fake vs True News Distribution")
plt.show()


# 3. Prepare data

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 4. Train Model

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


# 5. Graph: Train vs Test Accuracy

train_pred = model.predict(X_train_vec)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6,4))
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_acc, test_acc], color=['blue','green'])
plt.ylim(0,1)
plt.title("Training vs Testing Accuracy")
plt.ylabel("Accuracy")
plt.show()


# 6. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 7. Graph: Fake vs Real Predictions

fake_pred = list(y_pred).count(0)
true_pred = list(y_pred).count(1)

plt.figure(figsize=(6,4))
plt.bar(['Fake Predictions','Real Predictions'], [fake_pred, true_pred], color=['red','green'])
plt.title("Fake vs Real Predictions Count")
plt.ylabel("Count")
plt.show()


# 8. Classification Report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 9. Save model

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved as model.pkl & vectorizer.pkl")


# 10. Custom Prediction

input_news = ["India won the cricket world cup yesterday."]
input_vec = vectorizer.transform(input_news)
prediction = model.predict(input_vec)

if prediction[0] == 1:
    print("\nPrediction: Real News")
else:
    print("\nPrediction: Fake News")
