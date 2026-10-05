import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\data.csv")

print("\nDataset:")
print(df.head())

df = df.drop(columns=["Unnamed: 32"], errors="ignore")

X = df.drop(columns=["diagnosis", "id"], errors="ignore")
y = df["diagnosis"]

y = y.map({"M": 1, "B": 0})

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameters = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    SVC(),
    parameters,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

model = grid.best_estimator_

y_pred = model.predict(X_test_scaled)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

plt.figure()

plt.imshow(cm)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()
