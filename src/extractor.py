"""Data extraction module for fetching stock data from yfinance."""

import logging

import pandas as pd
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw ticker DataFrame: extract price, calculate returns, normalise dates.

    Args:
        df: Raw DataFrame from yfinance with date index and 'Close' column

    Returns:
        Processed DataFrame with 'Price' and 'Returns' columns and date index
    """
    # Keep only relevant column which is the close price
    df = df[["Close"]].rename(columns={"Close": "Price"})

    # Compute daily returns and drop the first date
    df["Returns"] = df["Price"].pct_change()
    df = df.dropna()

    # Convert index to date type
    df.index = df.index.date
    df.index.name = "Date"

    return df


def _extract_single_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Extract and process data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data download (YYYY-MM-DD format)
        end_date: End date for data download (YYYY-MM-DD format)

    Returns:
        Processed DataFrame or None if extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df_processed = _process_ticker_dataframe(df)

        if df.empty:
            logger.warning(f"No data available for ticker: {ticker}")
            return None

        return df_processed

    # Exception if ticker doesn't exist
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Extract historical stock data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for data download (YYYY-MM-DD format)
        end_date: End date for data download (YYYY-MM-DD format)

    Returns:
        Dictionary mapping ticker to DataFrame with columns ['Price', 'Returns']
    """
    all_stock_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df_processed = _extract_single_ticker_data(ticker, start_date, end_date)
        if df_processed is not None:
            all_stock_data[ticker] = df_processed

    return all_stock_data
