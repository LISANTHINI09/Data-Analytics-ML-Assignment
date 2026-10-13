# Program 4: SVM and Decision Tree classification (Iris dataset)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
 
iris = load_iris()
X, y = iris.data, iris.target
print("Features:", iris.feature_names)
print("Classes :", list(iris.target_names))
print("Shape   :", X.shape)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
 
# ---- SVM ----
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\n===== SVM (RBF kernel) =====")
print("Accuracy:", round(accuracy_score(y_test, svm_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=iris.target_names))
 
# ---- Decision Tree ----
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("\n===== Decision Tree (max_depth=3) =====")
print("Accuracy:", round(accuracy_score(y_test, dt_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=iris.target_names))
print("Feature importances:")
for name, imp in zip(iris.feature_names, dt.feature_importances_):
    print(f"  {name}: {imp:.4f}")
 
# Plot: decision tree
plt.figure(figsize=(12, 7))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree (Iris)")
plt.tight_layout()
plt.savefig("img/p4_decision_tree.png", dpi=150)
 
# Plot: SVM decision boundary using 2 features (petal length, petal width)
X2 = X[:, 2:4]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42, stratify=y)
svm2 = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X2_train, y2_train)
xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 300),
                     np.linspace(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, 300))
Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k", cmap="viridis")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("SVM Decision Boundary (RBF kernel)")
plt.tight_layout()
plt.savefig("img/p4_svm_boundary.png", dpi=150)
 
# Plot: confusion matrices side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, svm_pred), display_labels=iris.target_names).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("SVM")
ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred), display_labels=iris.target_names).plot(ax=ax[1], cmap="Greens", colorbar=False)
ax[1].set_title("Decision Tree")
plt.tight_layout()
plt.savefig("img/p4_confusion_matrices.png", dpi=150)
