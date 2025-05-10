import numpy as np

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from data import *

def main():
    np.random.seed(1234)

    train_data, test_data = load_S_Ar_R_As_data()

    dt = DecisionTree({"depth": 100})
    dt.train(*train_data)
    dt.evaluate(*train_data)
    dt.evaluate(*test_data)

    rf = RandomForest({"ntrees": 100, "feature_subset": 4, "depth": 20})
    rf.train(*train_data)
    rf.evaluate(*train_data)
    rf.evaluate(*test_data)

if __name__=="__main__":
    main()