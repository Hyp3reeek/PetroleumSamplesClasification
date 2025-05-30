from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.knn_classifier import KNNClassifier
from models.svm_classifier import SVMClassifier
from models.neural_net_classifier import NeuralNetClassifier
from itertools import combinations
from data import *
from helper import *


def main():
    np.random.seed(1234)

    features = ["Density", "S", "Ar", "R", "As"]
    for k in range(1, len(features) + 1):
        for feature_subset in combinations(features, k):
            print(f"Using features: {feature_subset}")
            train_data, test_data = load_feature_data(list(feature_subset))

            accuracies = {}

            # Decision Tree
            dt = DecisionTree({"depth": 10})
            dt.train(*train_data)
            dt.evaluate(*test_data)
            dt_acc = np.mean(dt.predict(test_data[0]) == test_data[1])
            accuracies["Decision Tree"] = dt_acc

            # Random Forest
            rf = RandomForest({"ntrees": 10, "feature_subset": k, "depth": 20})
            rf.train(*train_data)
            rf.evaluate(*test_data)
            rf_acc = np.mean(rf.predict(test_data[0]) == test_data[1])
            accuracies["Random Forest"] = rf_acc

            # KNN Classifier
            knn = KNNClassifier({"k": 3, "distance": "euclidean"})
            knn.train(*train_data)
            knn.evaluate(*test_data)
            kn_acc = np.mean(knn.predict(test_data[0]) == test_data[1])
            accuracies["KNN"] = kn_acc

            # SVM Classifier
            svm = SVMClassifier({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
            svm.train(*train_data)
            svm.evaluate(*test_data)
            svm_acc = np.mean(svm.predict(test_data[0]) == test_data[1])
            accuracies["SVM"] = svm_acc


            train_neural, test_neural = load_feature_data(list(feature_subset), load_data=load_neural_net_data)
            # Neural Network
            nnc = NeuralNetClassifier({"epochs": 2000, "lr": 0.01, "hidden_layers": [16, 16]})
            nnc.train(*train_neural)
            nnc.evaluate(*test_neural)
            nnc_acc = np.mean(nnc.predict(test_data[0]) == test_data[1])
            accuracies["NeuralNet"] = nnc_acc

            # Accuracy comparison
            # plot_accuracies(feature_subset, accuracies)

            # Combine data for projections
            x_combined = np.vstack((train_data[0], test_data[0]))
            y_combined = np.concatenate((train_data[1], test_data[1]))

            # PCA projection (unsupervised)
            # pca_projection(x_combined, y_combined, title=f"PCA Projection {feature_subset}")

            # UMAP projection (unsupervised)
            # plot_umap_projection(x_combined, y_combined, f"Features: {feature_subset}")

            # # LDA projection (supervised)
            # plot_lda_projection(x_combined, y_combined, f"Features: {feature_subset}")

            # # t-SNE projection (unsupervised)
            # plot_tsne_projection(x_combined, y_combined, f"Features: {feature_subset}")

            # # Decision boundary
            # if len(feature_subset) == 2:
            #     # svm = SVMClassifier({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
            #     # svm.train(*train_data)
            #     plot_decision_boundary(svm.model, test_data[0], test_data[1], f"SVM - {feature_subset}")

if __name__ == "__main__":
    main()