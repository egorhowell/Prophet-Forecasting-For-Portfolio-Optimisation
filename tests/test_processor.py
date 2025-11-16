"""Tests for processor module."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.processor import append_predictions, collect_recent_prices, preprocess_data


class TestProcessor:
    """Test data processing utilities."""

    def test_preprocess_data(self) -> None:
        """Test preprocessing data - aligns dates across tickers."""
        # Create sample data with overlapping dates
        dates1 = pd.date_range("2024-01-01", periods=10, freq="D")
        dates2 = pd.date_range("2024-01-03", periods=8, freq="D")  # Partial overlap

        data1 = pd.DataFrame(
            {
                "Price": np.random.randn(10) * 10 + 100,
                "Returns": np.random.randn(10) * 0.01,
            },
            index=dates1,
        )
        data2 = pd.DataFrame(
            {
                "Price": np.random.randn(8) * 10 + 50,
                "Returns": np.random.randn(8) * 0.02,
            },
            index=dates2,
        )

        data_dict = {"TICKER1": data1, "TICKER2": data2}
        aligned = preprocess_data(data_dict)

        assert isinstance(aligned, dict)
        assert len(aligned) == 2
        assert "TICKER1" in aligned
        assert "TICKER2" in aligned

        # Check that both DataFrames have the same index (common dates)
        assert list(aligned["TICKER1"].index) == list(aligned["TICKER2"].index)

        # Check that indexes are date type
        assert all(isinstance(d, date) for d in aligned["TICKER1"].index)
        assert all(isinstance(d, date) for d in aligned["TICKER2"].index)

        # Check columns are preserved
        assert "Price" in aligned["TICKER1"].columns
        assert "Returns" in aligned["TICKER1"].columns
        assert "Price" in aligned["TICKER2"].columns
        assert "Returns" in aligned["TICKER2"].columns

        # Common dates should be intersection of both ranges
        dates1_set = {d.date() for d in dates1}
        dates2_set = {d.date() for d in dates2}
        expected_common_dates = sorted(dates1_set & dates2_set)
        assert len(aligned["TICKER1"]) == len(expected_common_dates)

    def test_append_predictions(self) -> None:
        """Test appending predictions to portfolio data."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df1 = pd.DataFrame(
            {
                "Price": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Returns": [0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": [50.0, 51.0, 52.0, 53.0, 54.0],
                "Returns": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df1, "TICKER2": df2}
        predictions = {"TICKER1": 105.0, "TICKER2": 55.0}
        predicted_returns = {"TICKER1": 0.0096, "TICKER2": 0.0185}

        updated = append_predictions(portfolio_data, predictions, predicted_returns)

        assert isinstance(updated, dict)
        assert len(updated) == 2

        # Check TICKER1
        assert len(updated["TICKER1"]) == 6  # Original 5 + 1 prediction
        assert updated["TICKER1"].iloc[-1]["Price"] == 105.0
        assert updated["TICKER1"].iloc[-1]["Returns"] == 0.0096
        # Check prediction date is next day
        last_date = dates[-1].date()
        prediction_date = updated["TICKER1"].index[-1]
        assert prediction_date == last_date + timedelta(days=1)

        # Check TICKER2
        assert len(updated["TICKER2"]) == 6
        assert updated["TICKER2"].iloc[-1]["Price"] == 55.0
        assert updated["TICKER2"].iloc[-1]["Returns"] == 0.0185

    def test_append_predictions_preserves_original_data(self) -> None:
        """Test that appending predictions doesn't modify original DataFrames."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Price": [100.0, 101.0, 102.0],
                "Returns": [0.01, 0.01, 0.01],
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df.copy()}
        original_length = len(df)

        updated = append_predictions(
            portfolio_data,
            {"TICKER1": 103.0},
            {"TICKER1": 0.01},
        )

        # Original DataFrame should be unchanged
        assert len(df) == original_length
        # Updated DataFrame should have one more row
        assert len(updated["TICKER1"]) == original_length + 1

    def test_collect_recent_prices(self) -> None:
        """Test collecting recent prices returns the expected trailing values."""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "Price": np.linspace(100, 140, num=40),
                "Returns": np.random.randn(40) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        portfolio_data = {"TICKER1": df}

        recent_prices = collect_recent_prices(portfolio_data, days=30)

        assert "TICKER1" in recent_prices
        assert isinstance(recent_prices["TICKER1"], list)
        # Expect roughly 31 values (including last day) for 30-day window
        assert len(recent_prices["TICKER1"]) >= 30
        # First value in recent list should match cutoff price
        expected_start_price = float(df.loc[dates[-31].date(), "Price"])
        assert recent_prices["TICKER1"][0] == expected_start_price
        # Last value should be most recent price
        assert recent_prices["TICKER1"][-1] == float(df.iloc[-1]["Price"])
