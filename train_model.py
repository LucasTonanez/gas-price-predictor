import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


DATA_PATH = "data/gulf_coast_gas_oil_prices.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])


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
    "oil_diff_from_20_day_avg",
]

target = "next_day_gas"


# --------------------------------------------------
# Separate historical labeled data from future row
# --------------------------------------------------

# Only rows where the future gasoline price is already known
# should be used for training/evaluation.
labeled_df = df.dropna(subset=[target]).copy()

X = labeled_df[features]
y = labeled_df[target]


# --------------------------------------------------
# Chronological train/test split
# --------------------------------------------------

split_index = int(len(labeled_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Dataset")
print("------------------------------")
print(f"Total labeled rows: {len(labeled_df):,}")
print(f"Training rows:      {len(X_train):,}")
print(f"Testing rows:       {len(X_test):,}")
print()


# --------------------------------------------------
# Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

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
print(f"MAE:  {linear_mae:.6f}")
print(f"MSE:  {linear_mse:.6f}")
print(f"RMSE: {linear_rmse:.6f}")
print()


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

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
print(f"MAE:  {forest_mae:.6f}")
print(f"MSE:  {forest_mse:.6f}")
print(f"RMSE: {forest_rmse:.6f}")
print()


# --------------------------------------------------
# Choose best model
# --------------------------------------------------

if linear_mae <= forest_mae:
    best_model = linear_model
    best_model_name = "Linear Regression"
    best_mae = linear_mae
    best_rmse = linear_rmse
else:
    best_model = forest_model
    best_model_name = "Random Forest"
    best_mae = forest_mae
    best_rmse = forest_rmse


print("Best Test Model")
print("------------------------------")
print(f"Model: {best_model_name}")
print(f"MAE:   {best_mae:.6f}")
print(f"RMSE:  {best_rmse:.6f}")
print()


# --------------------------------------------------
# Retrain the winning model on ALL known data
# --------------------------------------------------

best_model.fit(X, y)


# --------------------------------------------------
# Forecast next unknown gasoline observation
# --------------------------------------------------

latest_row = df.iloc[[-1]]

latest_features = latest_row[features]

future_prediction = best_model.predict(
    latest_features
)[0]

latest_date = latest_row.iloc[0]["date"]
latest_gas_price = latest_row.iloc[0]["gas_price"]


print("Next Observation Forecast")
print("------------------------------")
print(f"Latest known date:      {latest_date.date()}")
print(f"Latest gasoline price:  ${latest_gas_price:.4f}")
print(f"Selected model:         {best_model_name}")
print(f"Predicted next price:   ${future_prediction:.4f}")
print(
    f"Predicted change:       "
    f"${future_prediction - latest_gas_price:+.4f}"
)
