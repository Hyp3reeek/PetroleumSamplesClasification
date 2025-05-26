from sklearn.svm import SVC
import numpy as np

class SVMClassifier:
    def __init__(self, params):
        self.kernel = params.get("kernel", "rbf")
        self.C = params.get("C", 1.0)
        self.gamma = params.get("gamma", "scale")
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        accuracy = np.mean(predicted == y)
        print(f"SVM Classifier accuracy: {round(accuracy, 2)}")
