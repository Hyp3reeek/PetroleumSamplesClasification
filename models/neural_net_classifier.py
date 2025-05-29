import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class TorchMultiLayerNetwork(nn.Module):
    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(n_i, n_o)
            for n_i, n_o in zip([n_in] + n_hiddens, n_hiddens + [n_out])
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return torch.sigmoid(self.layers[-1](x))


class NeuralNetClassifier:
    def __init__(self, params):
        self.params = params
        self.epochs = params.get("epochs", 1000)
        self.lr = params.get("lr", 0.01)
        self.hidden_layers = params.get("hidden_layers", [16, 16])
        self.model = None

    def train(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().view(-1, 1)

        self.model = TorchMultiLayerNetwork(X.shape[1], self.hidden_layers, 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = F.binary_cross_entropy(output, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                acc = ((output > 0.5).float() == y_tensor).float().mean()
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            preds = self.model(X_tensor)
        return (preds > 0.5).int().numpy().flatten()

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = np.mean(preds == y)
        print(f"Neural Network accuracy: {round(acc, 2)}")
