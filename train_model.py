import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/gulf_coast_gas_oil_prices.csv")

features = [
    "year",
    "month",
    "week_of_year",
    "day_of_week_num",

    "gas_lag_1",
    "gas_lag_2",
    "gas_lag_3",
    "gas_lag_5",
    "gas_lag_10",

    "gas_change_previous_day",
    "gas_change_previous_week",

    "gas_5_day_avg",
    "gas_10_day_avg",
    "gas_20_day_avg",

    "gas_diff_from_5_day_avg",
    "gas_diff_from_10_day_avg",
    "gas_diff_from_20_day_avg",

    "oil_lag_1",
    "oil_lag_2",
    "oil_lag_3",
    "oil_lag_5",
    "oil_lag_10",

    "oil_change_previous_day",
    "oil_change_previous_week",

    "oil_5_day_avg",
    "oil_10_day_avg",
    "oil_20_day_avg",

    "oil_diff_from_5_day_avg",
    "oil_diff_from_10_day_avg",
    "oil_diff_from_20_day_avg"
]

target = "next_day_gas"

X = df[features]
y = df[target]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = linear_mse ** 0.5

print("Linear Regression")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_mse = mean_squared_error(y_test, forest_predictions)
forest_rmse = forest_mse ** 0.5

print()
print("Random Forest")
print("MAE:", forest_mae)
print("MSE:", forest_mse)
print("RMSE:", forest_rmse)

if linear_mae < forest_mae:
    best_model = linear_model
    best_model_name = "Linear Regression"
else:
    best_model = forest_model
    best_model_name = "Random Forest"

latest_row = X.iloc[[-1]]
predicted_gas = best_model.predict(latest_row)[0]

print()
print("Best model:", best_model_name)
print("Latest known gas:", df.iloc[-1]["gas_price"])
print("Actual next day gas:", df.iloc[-1]["next_day_gas"])
print("Predicted next day gas:", predicted_gas)