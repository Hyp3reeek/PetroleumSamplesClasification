import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class TorchMultiLayerNetwork(nn.Module):
    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_i, n_o)
            for n_i, n_o in zip([n_in] + n_hiddens[:-1], n_hiddens)
        ])
        self.output_layer = nn.Linear(n_hiddens[-1], n_out)

    def forward(self, x):
        hidden_outputs = []
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            hidden_outputs.append(x)
        output = F.softmax(self.output_layer(x), dim=1)  # Multi-class
        return output, hidden_outputs


class NeuralNetClassifier:
    def __init__(self, params):
        self.params = params
        self.epochs = params.get("epochs", 1000)
        self.lr = params.get("lr", 0.01)
        self.hidden_layers = params.get("hidden_layers", [16, 16])
        self.n_classes = params.get("n_classes", 3)
        self.model = None

    def train(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()  # Multi-class labels

        self.model = TorchMultiLayerNetwork(X.shape[1], self.hidden_layers, self.n_classes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output, _ = self.model(X_tensor)
            loss = F.cross_entropy(output, y_tensor)  # Multi-class loss
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                preds = output.argmax(dim=1)
                acc = (preds == y_tensor).float().mean()
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            preds, _ = self.model(X_tensor)
        return preds.argmax(dim=1).numpy().flatten()

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = np.mean(preds == y)
        print(f"Neural Network accuracy: {round(acc, 4)}")
        return acc

    def get_hidden_outputs(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            _, hidden_outputs = self.model(X_tensor)
        return [h.numpy() for h in hidden_outputs]

    def visualize_decision_boundary(self, X, y, grid_step=0.01):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                             np.arange(y_min, y_max, grid_step))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.predict(grid)
        preds = preds.reshape(xx.shape)

        plt.contourf(xx, yy, preds, alpha=0.6, cmap=plt.cm.tab10)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.tab10)
        plt.title("Neural Network Decision Boundary (Multi-class)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def visualize_hidden_layers(self, X):
        hidden_outputs = self.get_hidden_outputs(X)

        fig, axes = plt.subplots(1, len(hidden_outputs), figsize=(15, 5))
        for i, hidden in enumerate(hidden_outputs):
            if hidden.shape[1] > 1:
                axes[i].scatter(hidden[:, 0], hidden[:, 1], alpha=0.5)
            else:
                axes[i].scatter(hidden[:, 0], np.zeros_like(hidden[:, 0]), alpha=0.5)
            axes[i].set_title(f'Hidden Layer {i+1}')
            axes[i].set_xlabel('Neuron 1')
            if hidden.shape[1] > 1:
                axes[i].set_ylabel('Neuron 2')
            else:
                axes[i].set_yticks([])
        plt.tight_layout()
        plt.show()
