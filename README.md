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

The prediction target is the change in gasoline price at the next available
observation:

```text
next_day_gas - gas_price
```

The latest feature-complete observation is kept even though its target is not
yet known, so it can be used for a genuine next-observation forecast.

## Models

Four forecasting approaches are compared:

### Persistence (Zero-Change) Baseline
Predicts that the next observed gasoline price will equal the current price,
which is equivalent to predicting a target change of zero.

### Linear Regression
Estimates a linear relationship between the engineered features and the next
observed price change.

### Ridge Regression
Fits a regularized linear model. Its feature scaler is fitted only on the
training partition to prevent information from the test period leaking into
training.

### Random Forest Regressor
Uses an ensemble of decision trees to capture nonlinear relationships between the engineered time-series features.

The method with the lowest test Mean Absolute Error (MAE), including the
persistence baseline, is selected for the final prediction.

## Evaluation

Because this is time-series data, the dataset is kept in chronological order rather than randomly shuffled.

- First 80% of observations → Training
- Final 20% → Testing

Models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

MAE and RMSE are measured in dollars per gallon. Because MSE squares the
errors, it is measured in (dollars per gallon)².

No shuffling is used. This prevents future observations from appearing in the
training set for earlier predictions.

The same chronological holdout is used both to compare methods and to select
the winner, so these results are not an independent final evaluation of the
selected method.

### Latest Results

Using FRED data downloaded on September 1, 2026, the labeled observations were
split as follows:

- Training: 2,312 rows, February 2, 2015 through April 25, 2024
- Testing: 579 rows, April 26, 2024 through August 24, 2026

Held-out metrics for the next-observation price change use dollars per gallon
for MAE and RMSE, and (dollars per gallon)² for MSE:

| Method | MAE | MSE | RMSE |
| --- | ---: | ---: | ---: |
| Persistence (zero change) | 0.045893 | 0.004453 | 0.066734 |
| Linear Regression | 0.046758 | 0.004595 | 0.067787 |
| Ridge Regression | 0.046739 | 0.004591 | 0.067754 |
| Random Forest | 0.048677 | 0.005079 | 0.071266 |

The persistence baseline had the lowest held-out MAE. None of the three
machine-learning models beat it on this split. Consequently, the forecast for
the latest available observation (August 25, 2026, at $3.4290 per gallon) is a
zero-change prediction of $3.4290 for the next available observation.

## Project Structure

```text
gas-price-predictor/
├── load_gas_data_api.py   # Downloads and processes FRED data
├── train_model.py         # Trains and evaluates regression models
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup
1. Clone the repository
git clone https://github.com/LucasTonanez/gas-price-predictor.git
cd gas-price-predictor
2. Install dependencies
pip install -r requirements.txt
3. Create a FRED API key
Create a .env file in the project root:
FRED_API_KEY=your_api_key_here
4. Create the data directory
mkdir data
5. Download and engineer the data
python load_gas_data_api.py
This creates:
data/gulf_coast_gas_oil_prices.csv
6. Train and evaluate
python train_model.py
The script reports metrics for all four approaches, identifies the method with
the lowest test MAE, and prints its next-observation gasoline-price prediction.
## Tech Stack
- Python
- pandas
- scikit-learn
- FRED API
- Requests
- python-dotenv
## Future Improvements
- Time-series cross-validation / walk-forward validation
- Additional energy and macroeconomic indicators
- Hyperparameter tuning for Random Forest
- Feature-importance analysis
- Gradient-boosting models
- Prediction-interval / uncertainty estimates
- Interactive visualization dashboard
## Disclaimer
This project is an educational machine-learning experiment and is not intended for financial, commodities-trading, or investment decisions.
