import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\IRIS.csv")

print("\nDataset:")
print(df.head())

if "species" in df.columns:
    target = "species"
elif "Species" in df.columns:
    target = "Species"
else:
    target = df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X = X.select_dtypes(include=["number"])

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

weak_classifier = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

model = AdaBoostClassifier(
    estimator=weak_classifier,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot()

plt.title("AdaBoost Confusion Matrix")
plt.show()

errors = []

for prediction in model.staged_predict(X_test):
    error = 1 - accuracy_score(y_test, prediction)
    errors.append(error)

plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel("Number of Weak Classifiers")
plt.ylabel("Error Rate")
plt.title("AdaBoost Error Rate")
plt.grid()
plt.show()


#2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\IRIS.csv")

if "species" in df.columns:
    target = "species"
elif "Species" in df.columns:
    target = "Species"
else:
    target = df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X = X.select_dtypes(include=["number"])

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

weak_classifier = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

weak_classifier.fit(X_train, y_train)

weak_pred = weak_classifier.predict(X_test)

weak_accuracy = accuracy_score(
    y_test,
    weak_pred
)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost.fit(X_train, y_train)

ada_pred = adaboost.predict(X_test)

ada_accuracy = accuracy_score(
    y_test,
    ada_pred
)

print("\nWeak Classifier Accuracy:",
      weak_accuracy * 100, "%")

print("AdaBoost Accuracy:",
      ada_accuracy * 100, "%")

models = [
    "Weak Classifier",
    "AdaBoost"
]

accuracies = [
    weak_accuracy * 100,
    ada_accuracy * 100
]

plt.figure(figsize=(12, 6))

bars = plt.bar(
    models,
    accuracies
)

plt.title("Weak Classifier vs AdaBoost")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.show()
