import os
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if api_key is None:
    raise ValueError("FRED_API_KEY was not found. Check your .env file.")


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def load_fred_series(series_id, column_name):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    response = requests.get(FRED_URL, params=params)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["observations"])
    df = df[["date", "value"]]
    df = df.rename(columns={"value": column_name})

    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(
        df[column_name],
        errors="coerce",
    )

    df = df.dropna()

    return df


# --------------------------------------------------
# Load FRED data
# --------------------------------------------------

gas_df = load_fred_series(
    "DGASUSGULF",
    "gas_price",
)

oil_df = load_fred_series(
    "DCOILWTICO",
    "oil_price",
)

df = pd.merge(
    gas_df,
    oil_df,
    on="date",
    how="inner",
)

df = df[df["date"].dt.year >= 2015].copy()


# --------------------------------------------------
# Calendar features
# --------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week_of_year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)
df["day_of_week"] = df["date"].dt.day_name()
df["day_of_week_num"] = df["date"].dt.dayofweek


# --------------------------------------------------
# Gasoline features
# --------------------------------------------------

df["gas_lag_1"] = df["gas_price"].shift(1)
df["gas_lag_2"] = df["gas_price"].shift(2)
df["gas_lag_3"] = df["gas_price"].shift(3)
df["gas_lag_5"] = df["gas_price"].shift(5)
df["gas_lag_10"] = df["gas_price"].shift(10)

df["gas_change_previous_day"] = (
    df["gas_lag_1"] - df["gas_lag_2"]
)

df["gas_change_previous_week"] = (
    df["gas_lag_1"] - df["gas_lag_5"]
)

df["gas_5_day_avg"] = (
    df["gas_price"]
    .shift(1)
    .rolling(window=5)
    .mean()
)

df["gas_10_day_avg"] = (
    df["gas_price"]
    .shift(1)
    .rolling(window=10)
    .mean()
)

df["gas_20_day_avg"] = (
    df["gas_price"]
    .shift(1)
    .rolling(window=20)
    .mean()
)

df["gas_diff_from_5_day_avg"] = (
    df["gas_lag_1"] - df["gas_5_day_avg"]
)

df["gas_diff_from_10_day_avg"] = (
    df["gas_lag_1"] - df["gas_10_day_avg"]
)

df["gas_diff_from_20_day_avg"] = (
    df["gas_lag_1"] - df["gas_20_day_avg"]
)


# --------------------------------------------------
# Crude-oil features
# --------------------------------------------------

df["oil_lag_1"] = df["oil_price"].shift(1)
df["oil_lag_2"] = df["oil_price"].shift(2)
df["oil_lag_3"] = df["oil_price"].shift(3)
df["oil_lag_5"] = df["oil_price"].shift(5)
df["oil_lag_10"] = df["oil_price"].shift(10)

df["oil_change_previous_day"] = (
    df["oil_lag_1"] - df["oil_lag_2"]
)

df["oil_change_previous_week"] = (
    df["oil_lag_1"] - df["oil_lag_5"]
)

df["oil_5_day_avg"] = (
    df["oil_price"]
    .shift(1)
    .rolling(window=5)
    .mean()
)

df["oil_10_day_avg"] = (
    df["oil_price"]
    .shift(1)
    .rolling(window=10)
    .mean()
)

df["oil_20_day_avg"] = (
    df["oil_price"]
    .shift(1)
    .rolling(window=20)
    .mean()
)

df["oil_diff_from_5_day_avg"] = (
    df["oil_lag_1"] - df["oil_5_day_avg"]
)

df["oil_diff_from_10_day_avg"] = (
    df["oil_lag_1"] - df["oil_10_day_avg"]
)

df["oil_diff_from_20_day_avg"] = (
    df["oil_lag_1"] - df["oil_20_day_avg"]
)


# --------------------------------------------------
# Prediction target
# --------------------------------------------------

df["next_day_gas"] = df["gas_price"].shift(-1)


# --------------------------------------------------
# Model features
# --------------------------------------------------

feature_columns = [
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


# Only remove rows missing model FEATURES.
# Keep the newest row even though its future target is unknown.
df = df.dropna(
    subset=feature_columns
).copy()


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True,
)

output_path = (
    "data/gulf_coast_gas_oil_prices.csv"
)

df.to_csv(
    output_path,
    index=False,
)

print(
    f"Saved {len(df):,} rows to {output_path}"
)

print(
    f"Latest observation date: "
    f"{df.iloc[-1]['date']}"
)

print(
    f"Latest gasoline price: "
    f"${df.iloc[-1]['gas_price']:.4f}"
)