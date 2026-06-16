import pickle
import numpy as np
import pandas as pd

with open("spam_balance_model.pkl", "rb") as f:
    model_data = pickle.load(f)

vectorizer = model_data["vectorizer"]
weights = model_data["weights"]
bias = model_data["bias"]
threshold = 0.63 #55 nếu là mô hình balance


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def predict_proba(X):
    scores = np.dot(X, weights) + bias
    return sigmoid(scores)

def predict(X):
    return (predict_proba(X) > threshold).astype(int)

df = pd.read_csv("test_balance.csv")

X_text = df["text"].astype(str).values
y_test = df["spam"].astype(int).values

X_test = vectorizer.transform(X_text).toarray()

y_prob = predict_proba(X_test)
y_pred = (y_prob > threshold).astype(int)

print("Features:", len(weights))
print("TF-IDF shape:", X_test.shape)


accuracy = np.mean(y_pred == y_test)

tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))

precision = tp / (tp + fp + 1e-15)
recall = tp / (tp + fn + 1e-15)
f1 = 2 * precision * recall / (precision + recall + 1e-15)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(np.array([
    [tn, fp],
    [fn, tp]
]))


