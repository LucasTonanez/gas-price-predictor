import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


DATA_PATH = (
    "data/gulf_coast_gas_oil_prices.csv"
)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
)


features = [
    "year",
    "month",
    "week_of_year",
    "day_of_week_num",

    # Current observations
    "gas_price",
    "oil_price",

    # Gas history
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

    # Oil history
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
    "oil_diff_from_20_day_avg",
]

target = "next_day_gas"


# --------------------------------------------------
# Historical labeled data
# --------------------------------------------------

labeled_df = df.dropna(
    subset=[target]
).copy()

X = labeled_df[features]
y = labeled_df[target]


# --------------------------------------------------
# Chronological 80/20 split
# --------------------------------------------------

split_index = int(
    len(labeled_df) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Dataset")
print("------------------------------")
print(
    f"Total labeled rows: "
    f"{len(labeled_df):,}"
)
print(
    f"Training rows:      "
    f"{len(X_train):,}"
)
print(
    f"Testing rows:       "
    f"{len(X_test):,}"
)
print()


# --------------------------------------------------
# Persistence baseline
# --------------------------------------------------

baseline_predictions = (
    labeled_df["gas_price"]
    .iloc[split_index:]
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_predictions,
)

baseline_rmse = baseline_mse ** 0.5


print("Persistence Baseline")
print("------------------------------")
print(
    "Prediction: next price = current price"
)
print(
    f"MAE:  {baseline_mae:.6f}"
)
print(
    f"MSE:  {baseline_mse:.6f}"
)
print(
    f"RMSE: {baseline_rmse:.6f}"
)
print()


# --------------------------------------------------
# Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train,
)

linear_predictions = (
    linear_model.predict(X_test)
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions,
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions,
)

linear_rmse = linear_mse ** 0.5


print("Linear Regression")
print("------------------------------")
print(
    f"MAE:  {linear_mae:.6f}"
)
print(
    f"MSE:  {linear_mse:.6f}"
)
print(
    f"RMSE: {linear_rmse:.6f}"
)
print()


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

forest_model.fit(
    X_train,
    y_train,
)

forest_predictions = (
    forest_model.predict(X_test)
)

forest_mae = mean_absolute_error(
    y_test,
    forest_predictions,
)

forest_mse = mean_squared_error(
    y_test,
    forest_predictions,
)

forest_rmse = forest_mse ** 0.5


print("Random Forest")
print("------------------------------")
print(
    f"MAE:  {forest_mae:.6f}"
)
print(
    f"MSE:  {forest_mse:.6f}"
)
print(
    f"RMSE: {forest_rmse:.6f}"
)
print()


# --------------------------------------------------
# Choose best ML model
# --------------------------------------------------

if linear_mae <= forest_mae:
    best_model = linear_model
    best_model_name = (
        "Linear Regression"
    )
    best_mae = linear_mae
    best_rmse = linear_rmse

else:
    best_model = forest_model
    best_model_name = (
        "Random Forest"
    )
    best_mae = forest_mae
    best_rmse = forest_rmse


print("Best ML Model")
print("------------------------------")
print(
    f"Model: {best_model_name}"
)
print(
    f"MAE:   {best_mae:.6f}"
)
print(
    f"RMSE:  {best_rmse:.6f}"
)


# Compare against baseline
mae_improvement = (
    baseline_mae - best_mae
)

percent_improvement = (
    mae_improvement
    / baseline_mae
    * 100
)

print(
    f"Baseline MAE: {baseline_mae:.6f}"
)

print(
    f"MAE improvement vs baseline: "
    f"{mae_improvement:.6f}"
)

print(
    f"Percent improvement: "
    f"{percent_improvement:.2f}%"
)

print()


# --------------------------------------------------
# Retrain winner using all labeled data
# --------------------------------------------------

best_model.fit(
    X,
    y,
)


# --------------------------------------------------
# Forecast newest unknown observation
# --------------------------------------------------

latest_row = df.iloc[[-1]]

latest_features = (
    latest_row[features]
)

future_prediction = (
    best_model.predict(
        latest_features
    )[0]
)

latest_date = (
    latest_row.iloc[0]["date"]
)

latest_gas_price = (
    latest_row.iloc[0]["gas_price"]
)

predicted_change = (
    future_prediction
    - latest_gas_price
)


print("Next Observation Forecast")
print("------------------------------")

print(
    f"Latest known date:      "
    f"{latest_date.date()}"
)

print(
    f"Latest gasoline price:  "
    f"${latest_gas_price:.4f}"
)

print(
    f"Selected model:         "
    f"{best_model_name}"
)

print(
    f"Predicted next price:   "
    f"${future_prediction:.4f}"
)

print(
    f"Predicted change:       "
    f"${predicted_change:+.4f}"
)