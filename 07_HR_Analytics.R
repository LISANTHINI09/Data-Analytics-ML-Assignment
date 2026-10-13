# Program 8: HR analytics in R - forecast next month's demand for hourly employees
# install.packages(c("forecast", "ggplot2"))
library(forecast)
library(ggplot2)
 
set.seed(2026)
months <- 36
t <- 1:months
season <- c(5, 3, 2, 4, 6, 8, 10, 11, 7, 5, 9, 14)           # yearly seasonal pattern
demand <- round(220 + 2.2 * t + rep(season, 3) * 4 + rnorm(months, 0, 6))
 
hr <- data.frame(Month = seq(as.Date("2023-01-01"), by = "month", length.out = months),
                 HourlyEmployeesNeeded = demand)
print(head(hr, 10))
print(summary(hr$HourlyEmployeesNeeded))
 
ts_data <- ts(hr$HourlyEmployeesNeeded, start = c(2023, 1), frequency = 12)
 
# Train / test split (last 6 months as test)
train_ts <- window(ts_data, end = c(2025, 6))
test_ts  <- window(ts_data, start = c(2025, 7))
 
# Models
fit_lm  <- tslm(train_ts ~ trend + season)
fit_ets <- ets(train_ts)
fit_ari <- auto.arima(train_ts)
 
cat("\nAccuracy (test set):\n")
print(accuracy(forecast(fit_lm,  h = 6), test_ts))
print(accuracy(forecast(fit_ets, h = 6), test_ts))
print(accuracy(forecast(fit_ari, h = 6), test_ts))
 
# Final model on all data and forecast next month
final <- tslm(ts_data ~ trend + season)
fc <- forecast(final, h = 1, level = 95)
print(fc)
cat("\nPredicted demand for next month:", round(fc$mean[1]), "hourly employees\n")
 
png("p8_forecast.png", width = 900, height = 550)
autoplot(forecast(final, h = 6)) +
  ggtitle("Hourly employee demand forecast") + xlab("Year") + ylab("Employees needed")
dev.off()
Python version used to produce the output below
# Python stand-in run for Program 8 (R not available in this workspace).
# Same synthetic HR data design and same model idea as tslm(ts ~ trend + season) in the R script.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
rng = np.random.default_rng(2026)
months = 36
t = np.arange(1, months + 1)
season = np.array([5, 3, 2, 4, 6, 8, 10, 11, 7, 5, 9, 14])
demand = np.round(220 + 2.2 * t + np.tile(season, 3) * 4 + rng.normal(0, 6, months)).astype(int)
dates = pd.date_range("2023-01-01", periods=months, freq="MS")
hr = pd.DataFrame({"Month": dates, "HourlyEmployeesNeeded": demand})
print(hr.head(10).to_string(index=False))
print("\nSummary:")
print(hr["HourlyEmployeesNeeded"].describe().round(2))
 
# Month dummies for all months present, aligned
all_cols = [f"m_{i}" for i in range(2, 13)]
def build(df_dates, tt):
    X = pd.DataFrame({"trend": tt})
    for c in all_cols:
        X[c] = (pd.DatetimeIndex(df_dates).month == int(c.split("_")[1])).astype(float)
    return X
 
X_all = build(hr["Month"], t)
train_n = 30
X_tr, X_te = X_all.iloc[:train_n], X_all.iloc[train_n:]
y_tr, y_te = demand[:train_n], demand[train_n:]
 
model = LinearRegression().fit(X_tr, y_tr)
pred_te = model.predict(X_te)
print("\nTest set (last 6 months):")
print(pd.DataFrame({"Month": hr["Month"].iloc[train_n:].dt.strftime("%b %Y").values,
                    "Actual": y_te, "Predicted": pred_te.round(1)}).to_string(index=False))
print("MAE :", round(mean_absolute_error(y_te, pred_te), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_te, pred_te)), 3))
print("MAPE:", round(np.mean(np.abs((y_te - pred_te) / y_te)) * 100, 3), "%")
 
# Final model on all data, forecast next month (Jan 2026)
final = LinearRegression().fit(X_all, demand)
next_date = pd.Series([pd.Timestamp("2026-01-01")])
X_next = build(next_date, np.array([months + 1]))
fc = final.predict(X_next)[0]
resid = demand - final.predict(X_all)
sd = resid.std(ddof=X_all.shape[1] + 1)
print(f"\nPredicted demand for next month (Jan 2026): {round(fc)} hourly employees")
print(f"Approx. 95% interval: {round(fc - 1.96 * sd)} to {round(fc + 1.96 * sd)}")
 
# 6-month forecast for plot
fut_dates = pd.Series(pd.date_range("2026-01-01", periods=6, freq="MS"))
X_fut = build(fut_dates, np.arange(months + 1, months + 7))
fut = final.predict(X_fut)
 
plt.figure(figsize=(9, 5.5))
plt.plot(hr["Month"], demand, marker="o", label="Actual demand")
plt.plot(hr["Month"], final.predict(X_all), linestyle="--", color="orange", label="Fitted")
plt.plot(fut_dates, fut, marker="s", color="red", label="Forecast (next 6 months)")
plt.fill_between(fut_dates, fut - 1.96 * sd, fut + 1.96 * sd, color="red", alpha=0.15, label="95% interval")
plt.title("Hourly employee demand forecast")
plt.xlabel("Month")
plt.ylabel("Employees needed")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/p8_forecast.png", dpi=150)
