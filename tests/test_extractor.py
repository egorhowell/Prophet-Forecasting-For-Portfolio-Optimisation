"""Tests for extractor module."""

from datetime import date

import pandas as pd

from src.extractor import extract_data


class TestExtractor:
    """Test data extraction."""

    def test_extract_data(self) -> None:
        """Test extracting historical data."""
        tickers = ["MSFT", "AAPL"]
        data = extract_data(tickers, start_date="2024-01-01")

        assert isinstance(data, dict)
        assert len(data) > 0
        for ticker in tickers:
            if ticker in data:
                assert isinstance(data[ticker], pd.DataFrame)
                assert "Price" in data[ticker].columns
                assert "Returns" in data[ticker].columns
                assert data[ticker].index.name == "Date"
                # Check that index is date type
                assert all(isinstance(d, date) for d in data[ticker].index)

    def test_extract_data_with_end_date(self) -> None:
        """Test extracting data with end_date filter."""
        tickers = ["KO"]
        end_date = "2024-06-01"
        data = extract_data(tickers, start_date="2024-01-01", end_date=end_date)

        assert isinstance(data, dict)
        if tickers[0] in data:
            assert isinstance(data[tickers[0]], pd.DataFrame)
            # Check that all dates are <= end_date
            if len(data[tickers[0]]) > 0:
                assert all(
                    pd.Timestamp(d) <= pd.Timestamp(end_date) for d in data[tickers[0]].index
                )
