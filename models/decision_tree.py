from collections import defaultdict
import numpy as np
from models.node import Node

class DecisionTree:
    def __init__(self, params):
        self.root_node = Node()
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        self.root_node.train(X, y, self.params)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [int(p) for p in predicted]  # Możesz tu usunąć zaokrąglenie, bo już masz wartości klas
        accuracy = np.mean(predicted == y)
        print(f"Decision Tree accuracy: {round(accuracy, 2)}")

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction

