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
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

df["previous_price"] = df["price"].shift(1)
df["price_change_last_week"] = df["price"] - df["previous_price"]

df["rolling_4_week_avg"] = df["price"].rolling(window=4).mean()
df["rolling_8_week_avg"] = df["price"].rolling(window=8).mean()

df = df.dropna()

print(df.head())
print(df.tail())