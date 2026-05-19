import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FRED_API_KEY")

if api_key is None:
    raise ValueError("FRED_API_KEY was not found. Check your .env file.")

url = "https://api.stlouisfed.org/fred/series/observations"

params = {
    "series_id": "DGASUSGULF",
    "api_key": api_key,
    "file_type": "json"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["observations"])

df = df[["date", "value"]]
df = df.rename(columns={"value": "price"})

df["date"] = pd.to_datetime(df["date"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df.dropna()

df["year"] = df["date"].dt.year
df = df[df["year"] >= 2015]
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day_of_week"] = df["date"].dt.day_name()
df["day_of_week_num"] = df["date"].dt.dayofweek

df["price_lag_1"] = df["price"].shift(1)
df["price_lag_2"] = df["price"].shift(2)
df["price_lag_3"] = df["price"].shift(3)
df["price_lag_5"] = df["price"].shift(5)
df["price_lag_10"] = df["price"].shift(10)

df["price_change_previous_day"] = df["price_lag_1"] - df["price_lag_2"]
df["price_change_previous_week"] = df["price_lag_1"] - df["price_lag_5"]

df["rolling_5_day_avg"] = df["price"].shift(1).rolling(window=5).mean()
df["rolling_10_day_avg"] = df["price"].shift(1).rolling(window=10).mean()
df["rolling_20_day_avg"] = df["price"].shift(1).rolling(window=20).mean()

df["diff_from_5_day_avg"] = df["price_lag_1"] - df["rolling_5_day_avg"]
df["diff_from_10_day_avg"] = df["price_lag_1"] - df["rolling_10_day_avg"]
df["diff_from_20_day_avg"] = df["price_lag_1"] - df["rolling_20_day_avg"]

df["next_day_price"] = df["price"].shift(-1)

df = df.dropna()

df.to_csv("data/gulf_coast_gas_prices.csv", index=False)