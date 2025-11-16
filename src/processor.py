"""Data processing module for aligning, normalizing, and manipulating stock data."""

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_data(all_stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Align multiple tickers by common dates while keeping them as a dictionary of DataFrames.

    Args:
        all_stock_data: Dictionary mapping ticker to DataFrame with date index

    Returns:
        Dictionary of DataFrames where all tickers share the same index (common dates)
    """
    if not all_stock_data:
        return {}

    # Ensure all indexes are datetime.date type and find intersection of dates across all tickers
    normalised_all_stock_data = {}
    for ticker, df in all_stock_data.items():
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index).date
        normalised_all_stock_data[ticker] = df_copy

    # Find common dates across all tickers
    date_sets = [set(df.index) for df in normalised_all_stock_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    # Trim each DataFrame to the common dates
    aligned_all_stock_data = {
        ticker: df.loc[common_dates] for ticker, df in normalised_all_stock_data.items()
    }
    return aligned_all_stock_data


def append_predictions(
    portfolio_data: dict[str, pd.DataFrame],
    predictions: dict[str, float],
    predicted_returns: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """
    Append predicted price and return to each ticker's DataFrame.

    Args:
        portfolio_data: Dictionary of historical DataFrames per ticker
        predictions: Dictionary of predicted prices per ticker
        predicted_returns: Dictionary of predicted returns per ticker

    Returns:
        Updated dictionary with an additional row for each ticker
    """
    updated_portfolio_data = {}

    for ticker, df in portfolio_data.items():
        df_copy = df.copy()

        last_date = df_copy.index[-1]
        prediction_date = last_date + timedelta(days=1)

        # Create and append prediction row
        new_row = pd.DataFrame(
            {"Price": [predictions[ticker]], "Returns": [predicted_returns[ticker]]},
            index=[prediction_date],
        )
        df_copy = pd.concat([df_copy, new_row])

        updated_portfolio_data[ticker] = df_copy

    return updated_portfolio_data


def collect_recent_prices(
    portfolio_data: dict[str, pd.DataFrame],
    days: int = 30,
) -> dict[str, list[float]]:
    """
    Collect the most recent prices for each ticker over the given trailing window.

    Args:
        portfolio_data: Dictionary of historical DataFrames per ticker.
        days: Number of trailing days (inclusive) to include. Defaults to 30.

    Returns:
        Dictionary mapping ticker to a list of recent price floats ordered by date.
    """
    recent_prices: dict[str, list[float]] = {}

    for ticker, df in portfolio_data.items():
        if df.empty:
            recent_prices[ticker] = []
            continue

        last_date = df.index[-1]
        cutoff = last_date - timedelta(days=days)
        recent_series = df.loc[df.index >= cutoff, "Price"]
        recent_prices[ticker] = [float(value) for value in recent_series.tolist()]

    return recent_prices
