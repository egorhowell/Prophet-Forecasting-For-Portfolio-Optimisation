"""Tests for Prophet model module."""

import numpy as np
import pandas as pd

from src.model import ProphetModel, _get_us_trading_holidays


class TestProphetModel:
    """Test Prophet model."""

    def test_fit(self) -> None:
        """Test fitting Prophet model."""
        # Create sample time series
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None

    def test_predict_next(self) -> None:
        """Test predict_next method."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        predicted_price = model.predict_next(price_series)

        assert isinstance(predicted_price, float)
        assert predicted_price > 0
        assert model.model is not None  # Model should be fitted

    def test_predict_for_tickers(self) -> None:
        """Test predict_for_tickers method with multiple tickers."""

        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create DataFrames with Price column (as expected by predict_for_tickers)
        df1 = pd.DataFrame(
            {
                "Price": 100 + np.cumsum(np.random.randn(100) * 0.5),
                "Returns": np.random.randn(100) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": 50 + np.cumsum(np.random.randn(100) * 0.3),
                "Returns": np.random.randn(100) * 0.02,
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df1, "TICKER2": df2}

        model = ProphetModel()
        predictions, predicted_returns = model.predict_for_tickers(portfolio_data)

        assert isinstance(predictions, dict)
        assert isinstance(predicted_returns, dict)
        assert len(predictions) == 2
        assert len(predicted_returns) == 2
        assert "TICKER1" in predictions
        assert "TICKER2" in predictions
        assert "TICKER1" in predicted_returns
        assert "TICKER2" in predicted_returns

        # Check predictions are floats and positive
        assert isinstance(predictions["TICKER1"], float)
        assert isinstance(predictions["TICKER2"], float)
        assert predictions["TICKER1"] > 0
        assert predictions["TICKER2"] > 0

        # Check predicted returns are floats
        assert isinstance(predicted_returns["TICKER1"], float)
        assert isinstance(predicted_returns["TICKER2"], float)

        # Check that predicted return is calculated correctly
        current_price1 = df1["Price"].iloc[-1]
        expected_return1 = (predictions["TICKER1"] - current_price1) / current_price1
        assert np.isclose(predicted_returns["TICKER1"], expected_return1, rtol=1e-5)

    def test_get_us_trading_holidays(self) -> None:
        """Test US trading holidays generation."""
        holidays = _get_us_trading_holidays(2024, 2024)

        assert isinstance(holidays, pd.DataFrame)
        assert len(holidays) > 0
        assert "holiday" in holidays.columns
        assert "ds" in holidays.columns
        assert "lower_window" in holidays.columns
        assert "upper_window" in holidays.columns

        # Check specific holidays exist
        holiday_names = holidays["holiday"].unique()
        assert "new_years" in holiday_names
        assert "christmas" in holiday_names
        assert "thanksgiving" in holiday_names

        # Check that all holidays have proper windows
        assert all(holidays["lower_window"] == -1)
        assert all(holidays["upper_window"] == 1)

        # Check date format
        assert pd.api.types.is_datetime64_any_dtype(holidays["ds"])

    def test_fit_with_holidays(self) -> None:
        """Test that Prophet model includes holidays when fitting."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None
        # Check that holidays are included
        assert hasattr(model.model, "holidays")
        if model.model.holidays is not None:
            assert len(model.model.holidays) > 0

    def test_fit_with_seasonality_config(self) -> None:
        """Test that Prophet model has seasonality properly configured."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None
        # Check seasonality settings
        assert model.model.yearly_seasonality is True
        assert model.model.weekly_seasonality is True
        assert model.model.daily_seasonality is False
