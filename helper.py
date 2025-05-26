from data import *
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import umap


def load_feature_data(feature_names):
    (x_train, y_train), (x_test, y_test) = load_data()

    feature_indices = {
        "Density": 0,
        "S": 1,
        "Ar": 2,
        "R": 3,
        "As": 4
    }

    selected_indices = [feature_indices[name] for name in feature_names]
    x_train_quadruple = x_train[:, selected_indices]
    x_test_quadruple = x_test[:, selected_indices]

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
    lda = LinearDiscriminantAnalysis()
    try:
        x_lda = lda.fit_transform(X, y)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(x_lda[y == label, 0], x_lda[y == label, 1], label=f'Class {label}')
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
        x_tsne = TSNE(n_components=2, perplexity=5, n_iter=5000).fit_transform(X)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(x_tsne[y == label, 0], x_tsne[y == label, 1], label=f'Class {label}')
        plt.title(f"t-SNE Projection - {title}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"t-SNE plot skipped: {e}")

def plot_decision_boundary(model, x, y, title, method="lda"):
    if x.shape[1] > 2:
        try:
            if method == "lda":
                if len(np.unique(y)) < 2:
                    print("LDA requires at least two classes")
                    return
                reducer = LinearDiscriminantAnalysis(n_components=2)
                x_proj = reducer.fit_transform(x, y)
                proj_title = f"{title} (LDA projection)"
            else:
                reducer = TSNE(n_components=2, perplexity=5, n_iter=5000)
                x_proj = reducer.fit_transform(x)
                proj_title = f"{title} (t-SNE projection)"
        except Exception as e:
            print(f"Projection failed: {e}")
            return
    else:
        x_proj = x
        proj_title = title

    try:
        disp = DecisionBoundaryDisplay.from_estimator(model, x_proj, response_method="predict", alpha=0.3)
        scatter = plt.scatter(x_proj[:, 0], x_proj[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
        plt.title(f"Decision Boundary - {proj_title}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.show()
    except Exception as e:
        print(f"Decision boundary plot skipped: {e}")



def umap_projection(X, y, title):
    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=5, metric='euclidean')
        x_umap = reducer.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(x_umap[y == label, 0], x_umap[y == label, 1], label=f'Class {label}')
        plt.title(f"UMAP Projection - {title}")
        plt.xlabel("UMAP Dim 1")
        plt.ylabel("UMAP Dim 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("UMAP is not installed. Skipping UMAP projection.")
    except Exception as e:
        print(f"UMAP plot skipped: {e}")

def pca_projection(X, y, title, n_components=2):
    try:
        pca = PCA(n_components)
        x_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(x_pca[y == label, 0], x_pca[y == label, 1], label=f'Class {label}')
        plt.title(f"PCA Projection - {title}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"PCA plot skipped: {e}")