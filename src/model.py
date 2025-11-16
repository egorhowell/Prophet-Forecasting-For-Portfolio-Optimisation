"""Prophet model for one-step forward prediction."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal
from prophet import Prophet

from .settings import HOLIDAY_NAME_MAP, PROPHET_PARAMS

logger = logging.getLogger(__name__)


def _normalise_holiday_name(name: str) -> str:
    """Convert calendar holiday names into Prophet-friendly labels."""
    if mapped := HOLIDAY_NAME_MAP.get(name):
        return mapped
    cleaned = name.lower()
    for char in ("'", ",", ".", "’"):
        cleaned = cleaned.replace(char, "")
    cleaned = cleaned.replace("&", "and").replace("-", "_")
    cleaned = "_".join(segment for segment in cleaned.split() if segment)
    return cleaned.strip("_")


def _get_us_trading_holidays(start_year: int = 2020, end_year: int = 2030) -> pd.DataFrame:
    """
    Fetch US trading holidays using the official exchange calendar.

    Args:
        start_year: Start year for holiday list.
        end_year: End year for holiday list.

    Returns:
        DataFrame with columns: holiday, ds, lower_window, upper_window.
    """
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    start = pd.Timestamp(date(start_year, 1, 1))
    end = pd.Timestamp(date(end_year, 12, 31))

    calendar = mcal.get_calendar("XNYS")
    holidays: list[dict[str, pd.Timestamp]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()

    if getattr(calendar, "regular_holidays", None) is not None:
        for rule in calendar.regular_holidays.rules:
            name = _normalise_holiday_name(rule.name)
            for holiday_date in rule.dates(start, end):
                timestamp = pd.Timestamp(holiday_date)
                if timestamp.tz is not None:
                    timestamp = timestamp.tz_localize(None)
                timestamp = timestamp.normalize()
                key = (name, timestamp)
                if key in seen:
                    continue
                seen.add(key)
                holidays.append({"holiday": name, "ds": timestamp})

    for holiday_date in getattr(calendar, "adhoc_holidays", []):
        timestamp = pd.Timestamp(holiday_date)
        if timestamp.tz is not None:
            timestamp = timestamp.tz_localize(None)
        timestamp = timestamp.normalize()
        if not (start <= timestamp <= end):
            continue
        key = ("adhoc_holiday", timestamp)
        if key in seen:
            continue
        seen.add(key)
        holidays.append({"holiday": "adhoc_holiday", "ds": timestamp})

    if not holidays:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    holidays_df = pd.DataFrame(holidays).drop_duplicates(subset=["holiday", "ds"])
    if holidays_df.empty:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
    holidays_df = holidays_df.sort_values("ds").reset_index(drop=True)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1
    holidays_df["upper_window"] = 1

    return holidays_df


class ProphetModel:
    """Prophet model for forecasting stock prices."""

    def __init__(self) -> None:
        """Initialise Prophet model."""
        self.model: Prophet | None = None

    def fit(self, price_series: pd.Series) -> ProphetModel:
        """
        Fit Prophet model to price series with trading holidays.

        Args:
            price_series: Historical price series with datetime index

        Returns:
            Self (ProphetModel instance) for method chaining
        """
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})

        # Get trading holidays for the date range in the data
        start_date = price_series.index.min()
        end_date = price_series.index.max()

        if isinstance(start_date, date):
            start_year = start_date.year
            end_year = end_date.year
        else:
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year

        # Get holidays and filter to relevant date range
        holidays = _get_us_trading_holidays(start_year - 1, end_year + 1)
        holidays = holidays[
            (holidays["ds"] >= pd.to_datetime(start_date))
            & (holidays["ds"] <= pd.to_datetime(end_date))
        ]

        # Initialise Prophet with holidays and seasonality
        prophet_params = PROPHET_PARAMS.copy()

        if not holidays.empty:
            prophet_params["holidays"] = holidays
            logger.info(f"Using {len(holidays)} trading holidays for Prophet model")
        else:
            logger.warning("No holidays found for date range, using Prophet without holidays")

        self.model = Prophet(**prophet_params)
        self.model.fit(df)

        return self

    def predict_next(self, price_series: pd.Series) -> float:
        """
        Fit model and predict next day's price in one step.

        Args:
            price_series: Historical price series including current day

        Returns:
            Predicted price for next day
        """

        self.fit(price_series)

        # Get the last date from the series
        last_date = price_series.index[-1]

        # Create future dataframe with next day
        future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=2, freq="D")[1:]})

        # Make prediction
        if self.model is None:
            raise RuntimeError("Model not fitted")
        forecast = self.model.predict(future)

        return float(forecast["yhat"].iloc[0])

    def predict_for_tickers(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Predict prices and returns for multiple tickers.

        Args:
            portfolio_data: Dictionary mapping ticker to DataFrame with 'Price' column

        Returns:
            Tuple containing:
            - predictions: dict[str, float] mapping ticker to predicted price
            - predicted_returns: dict[str, float] mapping ticker to predicted return
        """
        predictions: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}
        current_prices: dict[str, float] = {}

        for ticker in portfolio_data.keys():
            # Get stock data
            df_stock = portfolio_data[ticker]

            # Get current price
            current_price = df_stock["Price"].iloc[-1]
            current_prices[ticker] = current_price

            # Predict next day price
            predicted_price = self.predict_next(df_stock["Price"])
            predictions[ticker] = predicted_price

            # Calculate predicted return
            daily_return = (predicted_price - current_price) / current_price
            predicted_returns[ticker] = daily_return

        return predictions, predicted_returns
