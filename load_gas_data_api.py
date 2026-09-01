import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if api_key is None:
    raise ValueError("FRED_API_KEY was not found. Check your .env file.")

url = "https://api.stlouisfed.org/fred/series/observations"

def load_fred_series(series_id, column_name):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["observations"])

    df = df[["date", "value"]]
    df = df.rename(columns={"value": column_name})

    df["date"] = pd.to_datetime(df["date"])
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    df = df.dropna()

    return df


gas_df = load_fred_series("DGASUSGULF", "gas_price")
oil_df = load_fred_series("DCOILWTICO", "oil_price")

df = pd.merge(gas_df, oil_df, on="date", how="inner")

df["year"] = df["date"].dt.year
df = df[df["year"] >= 2015]

df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day_of_week"] = df["date"].dt.day_name()
df["day_of_week_num"] = df["date"].dt.dayofweek

df["gas_lag_1"] = df["gas_price"].shift(1)
df["gas_lag_2"] = df["gas_price"].shift(2)
df["gas_lag_3"] = df["gas_price"].shift(3)
df["gas_lag_5"] = df["gas_price"].shift(5)
df["gas_lag_10"] = df["gas_price"].shift(10)

df["gas_change_previous_day"] = df["gas_lag_1"] - df["gas_lag_2"]
df["gas_change_previous_week"] = df["gas_lag_1"] - df["gas_lag_5"]

df["gas_5_day_avg"] = df["gas_price"].shift(1).rolling(window=5).mean()
df["gas_10_day_avg"] = df["gas_price"].shift(1).rolling(window=10).mean()
df["gas_20_day_avg"] = df["gas_price"].shift(1).rolling(window=20).mean()

df["gas_diff_from_5_day_avg"] = df["gas_lag_1"] - df["gas_5_day_avg"]
df["gas_diff_from_10_day_avg"] = df["gas_lag_1"] - df["gas_10_day_avg"]
df["gas_diff_from_20_day_avg"] = df["gas_lag_1"] - df["gas_20_day_avg"]

df["next_day_gas"] = df["gas_price"].shift(-1)

df["oil_lag_1"] = df["oil_price"].shift(1)
df["oil_lag_2"] = df["oil_price"].shift(2)
df["oil_lag_3"] = df["oil_price"].shift(3)
df["oil_lag_5"] = df["oil_price"].shift(5)
df["oil_lag_10"] = df["oil_price"].shift(10)

df["oil_change_previous_day"] = df["oil_lag_1"] - df["oil_lag_2"]
df["oil_change_previous_week"] = df["oil_lag_1"] - df["oil_lag_5"]

df["oil_5_day_avg"] = df["oil_price"].shift(1).rolling(window=5).mean()
df["oil_10_day_avg"] = df["oil_price"].shift(1).rolling(window=10).mean()
df["oil_20_day_avg"] = df["oil_price"].shift(1).rolling(window=20).mean()

df["oil_diff_from_5_day_avg"] = df["oil_lag_1"] - df["oil_5_day_avg"]
df["oil_diff_from_10_day_avg"] = df["oil_lag_1"] - df["oil_10_day_avg"]
df["oil_diff_from_20_day_avg"] = df["oil_lag_1"] - df["oil_20_day_avg"]

df = df.dropna()

os.makedirs("data", exist_ok=True)
df.to_csv("data/gulf_coast_gas_oil_prices.csv", index=False)
