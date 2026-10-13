# Program 5: Naive Bayes Algorithm (Gaussian Naive Bayes on the Wine dataset)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
 
wine = load_wine()
X, y = wine.data, wine.target
print("Dataset shape:", X.shape)
print("Classes:", list(wine.target_names))
print("Class counts:", np.bincount(y))
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
 
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
 
print("\nClass priors:", nb.class_prior_.round(4))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
 
scores = cross_val_score(GaussianNB(), X, y, cv=5)
print("5-fold CV accuracies:", scores.round(4))
print("Mean CV accuracy    :", round(scores.mean(), 4))
 
# Plot 1: confusion matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=wine.target_names).plot(cmap="Oranges")
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.savefig("img/p5_confusion_matrix.png", dpi=150)
 
# Plot 2: class-wise distribution of two features with predicted probability
proba = nb.predict_proba(X_test)
plt.figure(figsize=(7, 5))
for c, name in enumerate(wine.target_names):
    plt.hist(X[y == c, 0], bins=12, alpha=0.55, label=name)
plt.xlabel(wine.feature_names[0])
plt.ylabel("Frequency")
plt.title("Feature distribution per class (Gaussian assumption)")
plt.legend()
plt.tight_layout()
plt.savefig("img/p5_feature_distribution.png", dpi=150)
 
print("\nPredicted probabilities for first 3 test samples:")
print(pd.DataFrame(proba[:3], columns=wine.target_names).round(4))
