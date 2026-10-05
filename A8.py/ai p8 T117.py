import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\winemag-data_first150k.csv")

print("\nDataset:")
print(df.head())

df = df[["country", "points", "price"]].dropna()

top_countries = df["country"].value_counts().head(5).index
df = df[df["country"].isin(top_countries)]

X = df[["points", "price"]]
y = df["country"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

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

k_values = range(1, 16)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, predictions))

best_k = list(k_values)[np.argmax(accuracies)]

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nBest K:", best_k)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

plt.figure(figsize=(9, 5))
plt.plot(list(k_values), accuracies, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K-NN Accuracy for Different K Values")
plt.xticks(list(k_values))
plt.grid()
plt.show()

sample_size = min(1500, len(X_test))

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    X_test.iloc[:sample_size]["points"],
    X_test.iloc[:sample_size]["price"],
    c=y_pred[:sample_size],
    alpha=0.7
)

plt.xlabel("Wine Points")
plt.ylabel("Wine Price")
plt.title("K-NN Wine Country Classification")
plt.colorbar(scatter, label="Predicted Country Class")
plt.show()

print("\nCountry Classes:")

for i, country in enumerate(encoder.classes_):
    print(i, "=", country)
