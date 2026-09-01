# Gulf Coast Gasoline Price Predictor

Machine-learning pipeline for predicting the next observed daily U.S. Gulf Coast regular gasoline price using historical gasoline and West Texas Intermediate (WTI) crude oil price data.

The project automatically retrieves economic data from the Federal Reserve Economic Data (FRED) API, engineers time-series features, trains multiple regression models, and evaluates their ability to predict the next day's gasoline price.

## Data

Data is retrieved directly from the FRED API.

### Gasoline
- **Series:** DGASUSGULF
- **Description:** Conventional Gasoline Prices: U.S. Gulf Coast, Regular
- **Frequency:** Daily
- **Units:** Dollars per gallon

### Crude Oil
- **Series:** DCOILWTICO
- **Description:** Crude Oil Prices: West Texas Intermediate (WTI) — Cushing, Oklahoma
- **Frequency:** Daily
- **Units:** Dollars per barrel

The current pipeline uses observations from 2015 onward.

## Feature Engineering

The model uses historical gasoline and crude-oil behavior rather than only the current price.

Features include:

### Calendar Features
- Year
- Month
- Week of year
- Day of week

### Gasoline Features
- 1, 2, 3, 5, and 10-day lagged prices
- Previous-day price change
- Previous-week price change
- 5, 10, and 20-day rolling averages
- Distance from each rolling average

### Crude Oil Features
- 1, 2, 3, 5, and 10-day lagged prices
- Previous-day price change
- Previous-week price change
- 5, 10, and 20-day rolling averages
- Distance from each rolling average

The prediction target is the gasoline price at the next available observation.

## Models

Two regression approaches are currently compared:

### Linear Regression
Provides a simple baseline for estimating the relationship between historical gasoline/oil behavior and the next gasoline price.

### Random Forest Regressor
Uses an ensemble of decision trees to capture nonlinear relationships between the engineered time-series features.

The model with the lower test Mean Absolute Error (MAE) is selected for the final prediction.

## Evaluation

Because this is time-series data, the dataset is kept in chronological order rather than randomly shuffled.

- First 80% of observations → Training
- Final 20% → Testing

Models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

This avoids allowing future observations to appear in the training set for earlier predictions.

## Project Structure

```text
gas-price-predictor/
├── load_gas_data_api.py   # Downloads and processes FRED data
├── train_model.py         # Trains and evaluates regression models
├── requirements.txt
├── .gitignore
└── README.md
