import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load training data
train_data = pd.read_csv('data/data.csv')

# Classification based on all features
X_train_all = train_data[["Density", "S", "Ar", "As"]]
y_train = train_data["Class"]

clf_all = DecisionTreeClassifier(max_depth=10, random_state=42)
clf_all.fit(X_train_all, y_train)

# Classification based on Density only
X_train_density = train_data[["Density"]]

clf_density = DecisionTreeClassifier(max_depth=10, random_state=42)
clf_density.fit(X_train_density, y_train)

# Classification based on Density, Ar, and As
X_train_density_ar_as = train_data[["Density", "Ar", "As"]]

clf_density_ar_as = DecisionTreeClassifier(max_depth=10, random_state=42)
clf_density_ar_as.fit(X_train_density_ar_as, y_train)

# Load classification data
classification_data = pd.read_csv('data/data2.csv')

# Predict using all features
X_classify_all = classification_data[["Density", "S", "Ar", "As"]]
classification_data["Predicted_Class_All"] = clf_all.predict(X_classify_all)

# Predict using Density only
X_classify_density = classification_data[["Density"]]
classification_data["Predicted_Class_Density"] = clf_density.predict(X_classify_density)

# Predict using Density, Ar, and As
X_classify_density_ar_as = classification_data[["Density", "Ar", "As"]]
classification_data["Predicted_Class_Density_Ar_As"] = clf_density_ar_as.predict(X_classify_density_ar_as)

# Save the updated classification data
classification_data.to_csv('data/data2_classified.csv', index=False)