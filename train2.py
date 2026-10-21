import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_excel("extracted_features.xlsx")
X = df.drop(columns=["Image_Name", "Category", "Label"]).values
y = df["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    acc = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X_scaled, y, cv=cv, scoring="precision").mean()
    rec = cross_val_score(model, X_scaled, y, cv=cv, scoring="recall").mean()
    f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1").mean()
    results.append({
        "Algorithm": name,
        "Accuracy": f"{acc.mean():.2%}",
        "Std": f"+/-{acc.std():.2%}",
        "Precision": f"{prec:.2%}",
        "Recall": f"{rec:.2%}",
        "F1-Score": f"{f1:.2%}",
    })

df_results = pd.DataFrame(results).sort_values(
    "Accuracy", key=lambda x: x.str.rstrip("%").astype(float), ascending=False
)
print(df_results.to_string(index=False))

plt.figure(figsize=(10, 5))
accs = [float(x.strip("%")) for x in df_results["Accuracy"]]
plt.barh(df_results["Algorithm"], accs, color="#10b981")
plt.xlabel("Accuracy (%)")
plt.title("Model Comparison (with HOG)")
plt.xlim(0, 100)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
