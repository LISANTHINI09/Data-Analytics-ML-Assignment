# Program 3: Logistic Regression
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
 
# Dataset: hours studied vs pass (1) / fail (0)
rng = np.random.default_rng(42)
hours = np.round(rng.uniform(0.5, 10, 100), 1)
prob = 1 / (1 + np.exp(-(1.1 * hours - 5.5)))
passed = (rng.random(100) < prob).astype(int)
df = pd.DataFrame({"HoursStudied": hours, "Passed": passed})
 
print("First 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["Passed"].value_counts())
 
X = df[["HoursStudied"]]
y = df["Passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
print("\nIntercept:", round(model.intercept_[0], 4))
print("Coefficient:", round(model.coef_[0][0], 4))
 
y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
 
for h in [2, 5, 8]:
    p = model.predict_proba(pd.DataFrame({"HoursStudied": [h]}))[0][1]
    print(f"Probability of passing after {h} hours of study: {p:.3f}")
 
# Plot 1: sigmoid curve
xs = np.linspace(0, 11, 200)
ps = model.predict_proba(pd.DataFrame({"HoursStudied": xs}))[:, 1]
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color="blue", label="Train")
plt.scatter(X_test, y_test, color="green", label="Test")
plt.plot(xs, ps, color="red", linewidth=2, label="Sigmoid curve")
plt.axhline(0.5, color="gray", linestyle="--")
plt.title("Logistic Regression: Hours Studied vs Pass Probability")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/p3_logistic_curve.png", dpi=150)
 
# Plot 2: confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Fail", "Pass"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("img/p3_confusion_matrix.png", dpi=150)
