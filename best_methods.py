import pandas as pd

# Load classified data
data = pd.read_csv('data/data2_classified.csv')

# Rename columns
data.rename(columns={
    'ONE': 'TSI',
    'TWO': 'CII',
    'THREE': 'S_VALUE',
    'FOUR': 'P_VALUE',
    'FIVE': 'Y_POINT'
}, inplace=True)

# Map Predicted_Class to stability classes
class_mapping = {
    2: 'light',
    1: 'heavy',
    0: 'medium'
}

classification_methods = ['Predicted_Class_All', 'Predicted_Class_Density', 'Predicted_Class_Density_Ar_As']

columns_to_analyze = ['TSI', 'CII', 'S_VALUE', 'P_VALUE', 'Y_POINT']

for method in classification_methods:
    data['Predicted_Stability'] = data[method].map(class_mapping)
    grouped_data = data.groupby('Predicted_Stability')[columns_to_analyze].sum()
    max_columns = grouped_data.idxmax(axis=1)

    print(f"Method with the highest count for each stability class ({method}):")
    print(max_columns)