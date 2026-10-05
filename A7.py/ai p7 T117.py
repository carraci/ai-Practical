import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

print("Adarsh Shukla T117")

df = pd.read_csv(r"C:\Users\Adarsh\Downloads\diabetes997.csv")

print("\nDataset:")
print(df.head())

if "Outcome" in df.columns:
    target = "Outcome"
elif "outcome" in df.columns:
    target = "outcome"
elif "Diabetes" in df.columns:
    target = "Diabetes"
else:
    target = df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

if y.dtype == "object":
    y = pd.factorize(y)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print("\nClass Probabilities:")
for cls, prob in zip(model.classes_, model.class_prior_):
    print("Class", cls, ":", round(prob, 4))

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot()

plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    range(len(y_test)),
    np.array(y_test),
    label="Actual",
    marker="o"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted",
    marker="x"
)

plt.xlabel("Test Sample Number")
plt.ylabel("Class")
plt.yticks([0, 1])
plt.title("Actual vs Predicted - Gaussian Naive Bayes")
plt.legend()
plt.grid()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC Coordinates:")

for i in range(len(fpr)):
    print(
        "FPR =", round(fpr[i], 4),
        "TPR =", round(tpr[i], 4)
    )

print("\nROC AUC:", round(roc_auc, 4))

plt.figure(figsize=(10, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label="Gaussian Naive Bayes (AUC = %.2f)" % roc_auc
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gaussian Naive Bayes")

plt.grid()
plt.legend(loc="lower right")
plt.show()
