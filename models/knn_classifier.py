import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, params):
        self.k = params.get("k", 3)
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nn_indices = np.argsort(distances)[:self.k]
            nn_labels = self.y_train[nn_indices]
            most_common = Counter(nn_labels).most_common(1)[0][0]
            preds.append(most_common)
        return np.array(preds)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        accuracy = np.mean(predicted == y)
        print(f"KNN Classifier accuracy: {round(accuracy, 2)}")