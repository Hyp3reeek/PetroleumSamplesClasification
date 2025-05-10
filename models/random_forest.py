from collections import defaultdict
import numpy as np
from scipy.stats import mode
from models.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, params):
        self.forest = []
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X,y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        print(f"Random forest accuracy: {round(np.mean(predicted == y), 2)}")

    def predict(self, X):
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        tree_predictions = np.array(tree_predictions).T  # shape: (n_samples, n_trees)
        forest_predictions = mode(tree_predictions, axis=1).mode.flatten()
        return forest_predictions

    def bagging(self, X, y):
        X_selected, y_selected = None, None
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_selected, y_selected = X[idx], y[idx]

        return X_selected, y_selected