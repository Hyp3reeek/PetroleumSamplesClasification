import pandas as pd
from sympy.strategies.core import switch

# Załaduj dane (jeśli nie są jeszcze załadowane)
df = pd.read_csv("data/data2_classified.csv")

# Utwórz kolumnę wskazującą poprawność (jeśli którakolwiek z ONE-FIVE to 1)
df["correct"] = df[["ONE", "TWO", "THREE", "FOUR", "FIVE"]].max(axis=1)

# Lista metod
methods = ["TSI", "CII", "S_VALUE", "P_VALUE", "Y_POINT"]

# Wyniki
results = []

# Dla każdej metody i klasy zlicz
for method in methods:
    if method not in df.columns:
        continue  # pomiń, jeśli kolumna nie istnieje
    for cls in [0, 1, 2]:
        subset = df[df["Predicted_Class_All"] == cls]
        total = len(subset)
        if total == 0:
            continue
        column = None
        if method == "TSI":column = "ONE"
        elif method == "CII":column = "TWO"
        elif method == "S_VALUE":column = "THREE"
        elif method == "P_VALUE":column = "FOUR"
        elif method == "Y_POINT":column = "FIVE"
        ones = subset[column].sum()
        zeros = total - ones
        percent = (ones / total * 100)
        cl = None
        if cls == 0 : cl = "Medium"
        elif cls == 1 : cl = "Heavy"
        elif cls == 2 : cl = "Light"
        results.append({
            "Method": method,
            "Class": cl,
            "Total": total,
            "Correct ": int(ones),
            "Incorrect ": int(zeros),
            "Accuracy %": round(percent, 2) if percent is not None else "N/A"
        })

# Calculate overall percentage score for all classes for each method
for method in methods:
    if method not in df.columns:
        continue
    total = len(df)
    if total == 0:
        continue
    column = None
    if method == "TSI": column = "ONE"
    elif method == "CII": column = "TWO"
    elif method == "S_VALUE": column = "THREE"
    elif method == "P_VALUE": column = "FOUR"
    elif method == "Y_POINT": column = "FIVE"
    ones = df[column].sum()
    zeros = total - ones
    percent = (ones / total * 100) if zeros != 0 else None
    results.append({
        "Method": method,
        "Class": "ALL",
        "Total": total,
        "Correct ": int(ones),
        "Incorrect ": int(zeros),
        "Accuracy %": round(percent, 2) if percent is not None else "N/A"
    })

# Wyniki jako DataFrame
results_df = pd.DataFrame(results)

# Wyświetl dane
print(results_df)

# Best for each class
best_per_class = results_df[results_df["Class"] != "ALL"].copy()
best_per_class = best_per_class[pd.to_numeric(best_per_class["Accuracy %"], errors="coerce").notnull()]
best_per_class["Accuracy %"] = best_per_class["Accuracy %"].astype(float)
best_per_class = best_per_class.sort_values("Accuracy %", ascending=False).groupby("Class").first()
# Best overall
best_overall = results_df[results_df["Class"] == "ALL"].sort_values("Accuracy %", ascending=False).iloc[0]

print("Best method for each class:")
print(best_per_class[["Method", "Accuracy %"]])
print("\nBest overall method:")
print(best_overall[["Method", "Accuracy %"]])