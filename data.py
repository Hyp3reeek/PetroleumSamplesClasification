import pandas as pd
import numpy as np


def load_data():
    # Load the dataset
    data = pd.read_csv('data/data1.csv')
    data = data[["ID", "Density", "S", "Ar", "R", "As", "CII", "Class"]]
    data = data.dropna().reset_index(drop=True)
    test_idx = np.random.choice(range(data.shape[0]), round(0.8*data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :].reset_index(drop=True)
    data_train = data.iloc[~test_idx, :].reset_index(drop=True)
    return data_train, data_test


def load_S_Ar_R_As_data():
    data = pd.read_csv('data/data.csv')
    # Wybieramy tylko interesujące nas kolumny
    data = data[["S", "Ar", "R", "As", "Class"]]
    data = data.dropna().reset_index(drop=True)

    # Losowy podział danych: 80% trening, 20% test
    test_idx = np.random.choice(range(data.shape[0]), round(0.2 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)

    # X = cechy, y = klasy
    X_train = data_train.drop("Class", axis=1).to_numpy()
    y_train = data_train["Class"].to_numpy()
    X_test = data_test.drop("Class", axis=1).to_numpy()
    y_test = data_test["Class"].to_numpy()

    return (X_train, y_train), (X_test, y_test)
