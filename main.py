import numpy as np

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from data import *
from helper import *

def main():
    np.random.seed(1234)

    train_data, test_data = load_S_Ar_R_As_data()
    print("Using all features S, Ar, R, As")
    dt = DecisionTree({"depth": 10})
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)

    rf = RandomForest({"ntrees": 10, "feature_subset": 4, "depth": 20})
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)

    # Use one feature only
    train_data, test_data = load_single_feature_data()
    print("Using single feature Density")
    dt = DecisionTree({"depth": 10})
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)
    rf = RandomForest({"ntrees": 10, "feature_subset": 1, "depth": 20})
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)

    # Use two features
    train_data, test_data = load_double_feature_data(("S", "Density"))
    print("Using two features")
    dt = DecisionTree({"depth": 10})
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)
    rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 20})
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)


    train_data, test_data = load_triple_feature_data(("S", "As", "R"))
    print("Using three features")
    dt = DecisionTree({"depth": 10})
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)
    rf = RandomForest({"ntrees": 10, "feature_subset": 3, "depth": 20})
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)


if __name__=="__main__":
    main()