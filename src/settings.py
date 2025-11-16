"""Settings and constants for portfolio optimisation."""
from datetime import datetime

# Risk parameters
MINIMUM_ALLOCATION = 0.05  # Minimum allocation per asset (5%)
MAXIMUM_ALLOCATION = 1
RISK_AVERSION = 5

# Date defaults
START_DATE = "2024-01-01"  # Default start date for historical data
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Stock Allocation
PORTFOLIO_TICKERS = [
    "AMD",
    "MSFT",
    "AAPL",
    "TSLA",
    "AMZN",
    "NVDA",
    "META",
    "GOOG",
    "TSM",
    "JPM",
    "NFLX",
    "PLTR",
]

# Database
SUPABASE_TABLE_NAME = "stock_optimisation_store"

# Holiday name mapping for Prophet model
HOLIDAY_NAME_MAP = {
    "New Year's Day": "new_years",
    "Dr. Martin Luther King Jr. Day": "mlk_day",
    "Good Friday": "good_friday",
    "Memorial Day": "memorial_day",
    "July 4th": "independence_day",
    "Labor Day": "labor_day",
    "Thanksgiving": "thanksgiving",
    "Election Day": "election_day",
    "Veteran Day": "veterans_day",
    "Columbus Day": "columbus_day",
    "Christmas": "christmas",
    "Christmas Day": "christmas",
}

# Prophet model parameters
PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
}
