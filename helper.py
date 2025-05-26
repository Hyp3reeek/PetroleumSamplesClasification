from data import *
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay


def load_feature_data(feature_names):
    (X_train, y_train), (X_test, y_test) = load_data()

    feature_indices = {
        "Density": 0,
        "S": 1,
        "Ar": 2,
        "R": 3,
        "As": 4
    }

    selected_indices = [feature_indices[name] for name in feature_names]
    x_train_quadruple = X_train[:, selected_indices]
    x_test_quadruple = X_test[:, selected_indices]

    return (x_train_quadruple, y_train), (x_test_quadruple, y_test)

def plot_accuracies(feature_subset, accuracies):
    classifiers = list(accuracies.keys())
    values = list(accuracies.values())
    plt.figure()
    plt.bar(classifiers, values)
    plt.ylim(0, 1)
    plt.title(f"Accuracy for features: {feature_subset}")
    plt.ylabel("Accuracy")
    plt.xlabel("Classifier")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_lda_projection(X, y, title):
    if len(np.unique(y)) < 2:
        return  # LDA requires at least two classes
    lda = LinearDiscriminantAnalysis(n_components=2)
    try:
        X_lda = lda.fit_transform(X, y)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=f'Class {label}')
        plt.title(f"LDA Projection - {title}")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"LDA plot skipped: {e}")

def plot_tsne_projection(X, y, title):
    try:
        X_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(X)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=f'Class {label}')
        plt.title(f"t-SNE Projection - {title}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"t-SNE plot skipped: {e}")

def plot_decision_boundary(model, X, y, title):
    if X.shape[1] != 2:
        return  # Boundaries only possible for 2D features
    try:
        disp = DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
        plt.title(f"Decision Boundary - {title}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Decision boundary plot skipped: {e}")