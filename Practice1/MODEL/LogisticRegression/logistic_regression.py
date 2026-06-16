import math
import re
import numpy as np
import pandas as pd
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trainl2.log", encoding="utf-8")
    ]
)

logger = logging.getLogger()


class LogisticRegression:
    def __init__(self, learning_rate=0.05, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.lambda_ = 0.01
    #   Elastic Net
    #   self.lambda_l1 = 0.005
    #   self.lambda_l2 = 0.005

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = (-1/m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1-y_true) * np.log(1-y_pred + 1e-15))
            # L1 Regularization
            # reg_loss = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            # loss += reg_loss
            # L2 Regularization
        reg_loss = (self.lambda_ / (2 * m)) * np.sum(self.weights ** 2)
        loss += reg_loss
            # Elastic Net loss
            # l1_loss = (self.lambda_l1 / m) * np.sum(np.abs(self.weights))
            # l2_loss = (self.lambda_l2 / (2 * m)) * np.sum(self.weights ** 2)
            # loss += l1_loss + l2_loss
        return loss
    def fit(self, X, y, X_val=None, y_val=None):   
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iter):

            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        #   L1 gradient (subgradient)
        #   dw += (self.lambda_ / n_samples) * np.sign(self.weights)
        #   Elastic Net gradient
        #   dw += (self.lambda_l1 / n_samples) * np.sign(self.weights)
        #   dw += (self.lambda_l2 / n_samples) * self.weights

        #   L2 gradient
            dw += (self.lambda_ / n_samples) * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # LOG TRAIN + VAL
            if i % 100 == 0:

                train_loss = self.compute_loss(y, y_pred)

                if X_val is not None:
                    val_pred = self.sigmoid(np.dot(X_val, self.weights) + self.bias)
                    val_loss = self.compute_loss(y_val, val_pred)

                    logger.info(
                        f"Iter {i} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
                    )
                else:
                    logger.info(f"Iter {i} | train_loss={train_loss:.4f}")
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        return (self.predict_proba(X) > 0.63).astype(int)

class Metrics:
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    def precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp + 1e-15)   
    def recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn + 1e-15)
    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 1e-15)
    def confusion_matrix(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])
    

file = r"/Users/hoangphuc.nguyen/Desktop/ML_Practice1/Balance_Data/train_balance.csv"
df = pd.read_csv(file)
X_text = df["text"].astype(str).values
y = df["spam"].values.astype(int)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1,1)
)

X = vectorizer.fit_transform(X_text).toarray()
X_train = X
y_train = y
model = LogisticRegression(learning_rate=0.5, max_iter=10000) 
model.fit(X_train, y_train)
y_pred = model.predict(X_train)


proba = model.predict_proba(X_train)

print("Min probability :", proba.min())
print("Max probability :", proba.max())
print("Mean probability:", proba.mean())

print("Percentiles:")
print("50% =", np.percentile(proba, 50))
print("75% =", np.percentile(proba, 75))
print("90% =", np.percentile(proba, 90))
print("95% =", np.percentile(proba, 95))
print("99% =", np.percentile(proba, 99))
y_pred = model.predict(X_train)

metrics = Metrics()

print("TRAIN Accuracy:", metrics.accuracy(y_train, y_pred))
print("TRAIN F1:", metrics.f1_score(y_train, y_pred))
print("Confusion matrix:\n", metrics.confusion_matrix(y_train, y_pred))


import pickle

model_data = {
    "vectorizer": vectorizer,
    "weights": model.weights,
    "bias": model.bias,
    "threshold": 0.63
}
with open("spam_balance_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved: spam_balance_model.pkl")
