import pandas as pd
def get_SP500():
    df = pd.read_csv("data/data1.csv")
    return df