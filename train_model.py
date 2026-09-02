import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/gulf_coast_gas_oil_prices.csv"

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

target_price = "next_day_gas"


def calculate_metrics(actual, predicted):
    """Return error metrics for predicted next-observation price changes."""
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
        "rmse": mse ** 0.5,
    }


df = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

if df.empty or not df["date"].is_monotonic_increasing:
    raise ValueError("The input data must contain chronologically ordered rows.")

# The final row has current features but no future price. Exclude it only from
# supervised evaluation so it remains available for a genuine forecast below.
labeled_df = df.dropna(subset=[target_price]).copy()
if len(labeled_df) < 2:
    raise ValueError("At least two labeled rows are required for an 80/20 split.")

X = labeled_df[features]
y = labeled_df[target_price] - labeled_df["gas_price"]

# Slice in chronological order. All fitting, including Ridge scaling, uses only
# the first 80 percent; no shuffled or future rows enter the training data.
split_index = int(len(labeled_df) * 0.8)
if split_index == 0 or split_index == len(labeled_df):
    raise ValueError("The chronological split produced an empty partition.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Dataset")
print("-" * 72)
print(f"Target:              {target_price} - gas_price")
print(f"Total labeled rows:  {len(labeled_df):,}")
print(f"Training rows:       {len(X_train):,}")
print(f"Testing rows:        {len(X_test):,}")
print(
    f"Training dates:      {labeled_df['date'].iloc[0].date()} to "
    f"{labeled_df['date'].iloc[split_index - 1].date()}"
)
print(
    f"Testing dates:       {labeled_df['date'].iloc[split_index].date()} to "
    f"{labeled_df['date'].iloc[-1].date()}"
)
print()

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
}

# A persistence forecast predicts no change, so its target prediction is zero.
baseline_name = "Persistence (zero change)"
results = {
    baseline_name: calculate_metrics(y_test, [0.0] * len(y_test)),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = calculate_metrics(y_test, model.predict(X_test))

baseline_mae = results[baseline_name]["mae"]

print("Test Metrics for Next-Observation Change ($/gallon)")
print("-" * 72)
print(f"{'Model':<27} {'MAE':>10} {'MSE':>10} {'RMSE':>10} {'vs baseline':>12}")
for name, metrics in results.items():
    mae_difference = baseline_mae - metrics["mae"]
    comparison = "--" if name == baseline_name else f"{mae_difference:+.6f}"
    print(
        f"{name:<27} {metrics['mae']:>10.6f} {metrics['mse']:>10.6f} "
        f"{metrics['rmse']:>10.6f} {comparison:>12}"
    )
print("vs baseline is baseline MAE minus model MAE; positive values are better.")
print()

best_name = min(results, key=lambda name: results[name]["mae"])
best_metrics = results[best_name]

print("Lowest Test MAE")
print("-" * 72)
print(f"Method: {best_name}")
print(f"MAE:    {best_metrics['mae']:.6f}")
print(f"RMSE:   {best_metrics['rmse']:.6f}")
print()

latest_row = df.iloc[[-1]]
if pd.notna(latest_row.iloc[0][target_price]):
    raise ValueError("The latest row already has a target and is not a future forecast row.")

latest_gas_price = latest_row.iloc[0]["gas_price"]

if best_name == baseline_name:
    predicted_change = 0.0
else:
    best_model = models[best_name]
    best_model.fit(X, y)
    predicted_change = best_model.predict(latest_row[features])[0]

future_prediction = latest_gas_price + predicted_change

print("Next Observation Forecast")
print("-" * 72)
print(f"Latest known date:      {latest_row.iloc[0]['date'].date()}")
print(f"Latest gasoline price:  ${latest_gas_price:.4f}")
print(f"Selected method:        {best_name}")
print(f"Predicted next price:   ${future_prediction:.4f}")
print(f"Predicted change:       ${predicted_change:+.4f}")
