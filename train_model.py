import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/gulf_coast_gas_prices.csv")

features = [
    "year",
    "month",
    "week_of_year",
    "day_of_week_num",
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_5",
    "price_lag_10",
    "price_change_previous_day",
    "price_change_previous_week",
    "rolling_5_day_avg",
    "rolling_10_day_avg",
    "rolling_20_day_avg",
    "diff_from_5_day_avg",
    "diff_from_10_day_avg",
    "diff_from_20_day_avg"
]

target = "next_day_price"

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

print("Linear Regression")
print("MAE:", linear_mae)
print("MSE:", linear_mse)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_mse = mean_squared_error(y_test, forest_predictions)

print()
print("Random Forest")
print("MAE:", forest_mae)
print("MSE:", forest_mse)

latest_row = X.iloc[[-1]]
predicted_price = linear_model.predict(latest_row)[0]

if linear_mae < forest_mae:
    best_model = linear_model
    best_model_name = "Linear Regression"
else:
    best_model = forest_model
    best_model_name = "Random Forest"

latest_row = X.iloc[[-1]]
predicted_price = best_model.predict(latest_row)[0]

print()
print("Best model:", best_model_name)
print("Latest known price:", df.iloc[-1]["price"])
print("Actual next day price:", df.iloc[-1]["next_day_price"])
print("Predicted next day price:", predicted_price)