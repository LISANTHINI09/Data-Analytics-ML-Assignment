# Program 2: Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
 
# Dataset: years of experience vs salary (in thousands)
data = {
    "YearsExperience": [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    "Salary": [39.3, 46.2, 37.7, 43.5, 39.8, 56.6, 60.1, 54.4, 64.4, 57.2,
               63.2, 55.8, 56.9, 57.0, 61.1, 67.9, 66.0, 83.1, 81.4, 94.0,
               91.7, 98.3, 101.3, 113.8, 109.4, 105.6, 116.9, 112.6, 122.4, 121.9],
}
df = pd.DataFrame(data)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)
 
X = df[["YearsExperience"]]
y = df["Salary"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
print("\nIntercept (b0):", round(model.intercept_, 4))
print("Slope (b1):", round(model.coef_[0], 4))
print(f"Equation: Salary = {model.intercept_:.4f} + {model.coef_[0]:.4f} * YearsExperience")
 
y_pred = model.predict(X_test)
print("\nActual vs Predicted (test set):")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2)}))
 
print("\nMean Absolute Error :", round(mean_absolute_error(y_test, y_pred), 4))
print("Mean Squared Error  :", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE                :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R-squared (R2)      :", round(r2_score(y_test, y_pred), 4))
 
new_exp = pd.DataFrame({"YearsExperience": [12]})
print("\nPredicted salary for 12 years of experience:", round(model.predict(new_exp)[0], 2))
 
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color="blue", label="Training data")
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression line")
plt.title("Simple Linear Regression: Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/p2_linear_regression.png", dpi=150)
