import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def parse_int(val):
    if pd.isna(val) or val == '':
        return 0
    try:
        return int(float(str(val).replace(',', '.')))
    except ValueError:
        return 0

# 1. Wczytanie i przekształcenie pvalue.csv
df = pd.read_csv('data/1.csv')

# mapa p-value dla czystych próbek (100/0/0)
pure = df[(df['P1'] == 100) & (df['P2'] == 0) & (df['P3'] == 0)]
pure_map = dict(zip(pure['Sample1'], pure['P-ValueTotal']))

# wybieramy TYLKO mieszaniny (P2>0 lub P3>0)
mix = df[~((df['P1'] == 100) & (df['P2'] == 0) & (df['P3'] == 0))]

# rozbijamy na wiersze z kolumnami Sample1–3, P1–3, P-Value1–3, P-ValueTotal
rows = []
for _, r in mix.iterrows():
    rows.append({
        'Sample1': r['Sample1'],
        'Sample2': r['Sample2'] if pd.notna(r['Sample2']) else '',
        'Sample3': r['Sample3'] if pd.notna(r['Sample3']) else '',
        'P1': parse_int(r['P1']),
        'P2': parse_int(r['P2']),
        'P3': parse_int(r['P3']),
        'P-Value1': r['P-Value1'],
        'P-Value2': r['P-Value2'],
        'P-Value3': r['P-Value3'],
        'P-ValueTotal': r['P-ValueTotal']
    })

mixture_df = pd.DataFrame(rows, columns=[
    'Sample1','Sample2','Sample3','P1','P2','P3',
    'P-Value1','P-Value2','P-Value3','P-ValueTotal'
])

# 2. Wczytanie klasyfikacji pojedynczych próbek
class_df = pd.read_csv('data/data2_classified.csv')
class_df = class_df[['Name', 'Predicted_Class_All']] \
    .rename(columns={'Predicted_Class_All': 'Class'})

# Assign classes to Sample1 and Sample2, handling missing Sample2
mixture_df = mixture_df \
    .merge(class_df, left_on='Sample1', right_on='Name', how='left') \
    .rename(columns={'Class': 'Class1'}) \
    .drop(columns=['Name'])
mixture_df = mixture_df \
    .merge(class_df, left_on='Sample2', right_on='Name', how='left') \
    .rename(columns={'Class': 'Class2'}) \
    .drop(columns=['Name'])


# Zapis tabeli mieszanki do pliku CSV
mixture_df.to_csv('data/mixtures_long_pvalue.csv', index=False)

# jeżeli trzeciej próbki nie chcemy używać, możemy pominąć Sample3/P3

# 4. Przygotowanie cech i etykiety
#    tu bierzemy tylko dwuskładnikowe mieszanki (Sample3=='')
mixture_df = mixture_df[mixture_df['Sample2'] != '']

X = mixture_df[['Class1', 'Class2', 'P1', 'P2']]
y = mixture_df['P-ValueTotal']

# 5. Trenowanie i ocena
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
mse = -scores.mean()
print(f"5-fold CV MSE = {mse:.3f}")

rf.fit(X, y)

# 6. Zapis modelu
import os
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/regressor_pvalue.joblib')
print("Model zapisany do models/regressor_pvalue.joblib")
