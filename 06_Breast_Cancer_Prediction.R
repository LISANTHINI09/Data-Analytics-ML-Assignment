# Program 7: Predictive analysis in R - diagnose breast cancer (Wisconsin data)
# install.packages(c("mlbench", "caret", "e1071", "rpart", "rpart.plot", "randomForest"))
library(mlbench)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
 
data(BreastCancer)
bc <- BreastCancer
bc$Id <- NULL
bc <- na.omit(bc)                       # remove rows with missing values
bc[, 1:9] <- lapply(bc[, 1:9], function(x) as.numeric(as.character(x)))
bc$Class <- factor(bc$Class, levels = c("benign", "malignant"))
 
cat("Dimensions:", dim(bc), "\n")
print(table(bc$Class))
print(summary(bc))
 
set.seed(123)
idx   <- createDataPartition(bc$Class, p = 0.7, list = FALSE)
train <- bc[idx, ]
test  <- bc[-idx, ]
 
# Logistic regression
log_model <- glm(Class ~ ., data = train, family = binomial)
log_prob  <- predict(log_model, test, type = "response")
log_pred  <- factor(ifelse(log_prob > 0.5, "malignant", "benign"), levels = levels(bc$Class))
cat("\n--- Logistic Regression ---\n")
print(confusionMatrix(log_pred, test$Class, positive = "malignant"))
 
# Decision tree
tree_model <- rpart(Class ~ ., data = train, method = "class")
png("p7_tree.png", width = 800, height = 600)
rpart.plot(tree_model, main = "Decision Tree - Breast Cancer")
dev.off()
tree_pred <- predict(tree_model, test, type = "class")
cat("\n--- Decision Tree ---\n")
print(confusionMatrix(tree_pred, test$Class, positive = "malignant"))
 
# Random forest
set.seed(123)
rf_model <- randomForest(Class ~ ., data = train, ntree = 300, importance = TRUE)
rf_pred  <- predict(rf_model, test)
cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, test$Class, positive = "malignant"))
png("p7_importance.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Variable Importance")
dev.off()
 
# Predict for a new patient
new_patient <- data.frame(Cl.thickness = 8, Cell.size = 7, Cell.shape = 7, Marg.adhesion = 5,
                          Epith.c.size = 6, Bare.nuclei = 10, Bl.cromatin = 7,
                          Normal.nucleoli = 8, Mitoses = 1)
cat("\nPrediction for new patient:", as.character(predict(rf_model, new_patient)), "\n")
Python version used to produce the output below
# Python stand-in run for Program 7 (R not available in this workspace).
# Uses the Wisconsin breast cancer data bundled with scikit-learn, same three models as the R script.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
 
d = load_breast_cancer()
X, y = d.data, d.target          # 0 = malignant, 1 = benign
print("Dimensions:", X.shape)
print("Class counts: malignant =", int((y == 0).sum()), ", benign =", int((y == 1).sum()))
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
names = ["malignant", "benign"]
 
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=123),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=123),
}
preds = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    preds[name] = p
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_test, p), 4))
    print("Confusion matrix (rows = actual, cols = predicted; order: malignant, benign):")
    print(confusion_matrix(y_test, p))
    print(classification_report(y_test, p, target_names=names))
 
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=d.feature_names).sort_values(ascending=False)
print("Top 10 important features (Random Forest):")
print(imp.head(10).round(4))
 
sample = X_test[0:1]
print("\nPrediction for first test patient:", names[int(rf.predict(sample)[0])], "| actual:", names[int(y_test[0])])
 
plt.figure(figsize=(14, 7))
plot_tree(models["Decision Tree"], feature_names=d.feature_names, class_names=names, filled=True, rounded=True, fontsize=7)
plt.title("Decision Tree - Breast Cancer")
plt.tight_layout()
plt.savefig("img/p7_tree.png", dpi=130)
 
plt.figure(figsize=(8, 6))
imp.head(10)[::-1].plot(kind="barh", color="teal")
plt.title("Variable Importance (Random Forest)")
plt.tight_layout()
plt.savefig("img/p7_importance.png", dpi=150)
 
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, p) in zip(ax, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p), display_labels=names).plot(ax=a, cmap="Blues", colorbar=False)
    a.set_title(name)
plt.tight_layout()
plt.savefig("img/p7_confusion_matrices.png", dpi=150)
