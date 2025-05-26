import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv('data/data1.csv')
    data = data[["Density", "S", "Ar", "R", "As", "Class"]]
    data = data.dropna().reset_index(drop=True)
    # Encode class labels as integers
    data["Class"] = data["Class"].astype("category").cat.codes
    test_idx = np.random.choice(range(data.shape[0]), round(0.2 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    x_train = data_train.drop("Class", axis=1).to_numpy()
    y_train = data_train["Class"].to_numpy()
    x_test = data_test.drop("Class", axis=1).to_numpy()
    y_test = data_test["Class"].to_numpy()
    return (x_train, y_train), (x_test, y_test)


# def load_S_Ar_R_As_data():
#     data = pd.read_csv('data/data.csv')
#     # Wybieramy tylko interesujące nas kolumny
#     data = data[["S", "Ar", "R", "As", "Class"]]
#     data = data.dropna().reset_index(drop=True)
#
#     # Losowy podział danych: 80% trening, 20% test
#     test_idx = np.random.choice(range(data.shape[0]), round(0.2 * data.shape[0]), replace=False)
#     data_test = data.iloc[test_idx, :]
#     data_train = data.drop(test_idx, axis=0)
#
#     # X = cechy, y = klasy
#     x_train = data_train.drop("Class", axis=1).to_numpy()
#     y_train = data_train["Class"].to_numpy()
#     x_test = data_test.drop("Class", axis=1).to_numpy()
#     y_test = data_test["Class"].to_numpy()
#
#     return (x_train, y_train), (x_test, y_test)
