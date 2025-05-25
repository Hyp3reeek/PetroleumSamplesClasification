from data import *


def load_single_feature_data(feature_name="Density"):
    (X_train, y_train), (X_test, y_test) = load_data()

    feature_index = {
        "Density": 0,
        "S": 1,
        "Ar": 2,
        "R": 3,
        "As": 4
    }[feature_name]

    x_train_single = X_train[:, [feature_index]]
    x_test_single = X_test[:, [feature_index]]

    return (x_train_single, y_train), (x_test_single, y_test)

def load_double_feature_data(feature_names=("Density", "S")):
    (X_train, y_train), (X_test, y_test) = load_data()

    feature_indices = {
        "Density": 0,
        "S": 1,
        "Ar": 2,
        "R": 3,
        "As": 4
    }

    selected_indices = [feature_indices[name] for name in feature_names]
    x_train_double = X_train[:, selected_indices]
    x_test_double = X_test[:, selected_indices]

    return (x_train_double, y_train), (x_test_double, y_test)

def load_triple_feature_data(feature_names=("Density", "S", "Ar")):
    (X_train, y_train), (X_test, y_test) = load_data()

    feature_indices = {
        "Density": 0,
        "S": 1,
        "Ar": 2,
        "R": 3,
        "As": 4
    }

    selected_indices = [feature_indices[name] for name in feature_names]
    x_train_triple = X_train[:, selected_indices]
    x_test_triple = X_test[:, selected_indices]

    return (x_train_triple, y_train), (x_test_triple, y_test)