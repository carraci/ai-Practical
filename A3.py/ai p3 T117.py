print("Adarsh Shukla T117")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\IRIS.csv")

print(df.columns.tolist())

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

target_column = df.columns[-1]

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

X = df[numeric_columns[:2]]
y = df[target_column]

print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

plt.figure(figsize=(14, 7))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(x) for x in model.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Using Two Features")
plt.tight_layout()
plt.show()
