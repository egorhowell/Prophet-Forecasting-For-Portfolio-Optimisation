"""Tests for database module."""

import json
import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.database import get_supabase_client, save_results_to_supabase
from src.settings import SUPABASE_TABLE_NAME


class TestGetSupabaseClient:
    """Test Supabase client creation."""

    @patch.dict(
        "os.environ", {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
    )
    @patch("src.database.create_client")
    def test_get_supabase_client_with_credentials(self, mock_create_client: MagicMock) -> None:
        """Test get_supabase_client returns client when credentials are available."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        result = get_supabase_client()

        assert result is not None
        assert result == mock_client
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_supabase_client_without_credentials(self) -> None:
        """Test get_supabase_client returns None when both credentials are missing."""
        result = get_supabase_client()
        assert result is None


class TestSaveResultsToSupabase:
    """Test saving results to Supabase."""

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_success(self, mock_get_client: MagicMock) -> None:
        """Test successfully saving results to Supabase."""
        # Setup mocks
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Test data
        result = {
            "date": date(2024, 1, 31),
            "predictions": {"AAPL": 150.25, "MSFT": 380.50},
            "predicted_returns": {"AAPL": 0.02, "MSFT": 0.015},
            "weights": {"AAPL": 0.4, "MSFT": 0.6},
            "actual_prices_last_month": {"AAPL": [148.0], "MSFT": [375.0]},
        }

        # Call function
        save_results_to_supabase(result)

        # Verify calls
        mock_get_client.assert_called_once()
        mock_client.table.assert_called_once_with(SUPABASE_TABLE_NAME)
        mock_table.insert.assert_called_once()

        # Check that insert was called with correct structure
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 2  # Two stocks

        # Check first row structure
        first_row = insert_call_args[0]
        assert "id" in first_row
        assert "created_at" in first_row
        assert "as_of_date" in first_row
        assert "ticker" in first_row
        assert "predicted_price" in first_row
        assert "predicted_return" in first_row
        assert "actual_prices_last_month" in first_row
        assert "portfolio_weight" in first_row

        # Check ID is a valid UUID string
        uuid.UUID(first_row["id"])  # Will raise if invalid UUID

        # Check values
        assert first_row["ticker"] in ("AAPL", "MSFT")
        assert first_row["as_of_date"] == "2024-01-31"
        if first_row["ticker"] == "AAPL":
            assert first_row["predicted_price"] == 150.25
            assert first_row["predicted_return"] == 0.02
            assert first_row["portfolio_weight"] == 0.4
            actuals = json.loads(first_row["actual_prices_last_month"])
            assert actuals == [148.0]
        else:
            assert first_row["predicted_price"] == 380.50
            assert first_row["predicted_return"] == 0.015
            assert first_row["portfolio_weight"] == 0.6
            actuals = json.loads(first_row["actual_prices_last_month"])
            assert actuals == [375.0]

        mock_insert.execute.assert_called_once()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_no_predictions(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase returns early when no predictions."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = {
            "date": date(2024, 1, 31),
            "predictions": {},
            "predicted_returns": {},
            "weights": {},
            "actual_prices_last_month": {},
        }

        save_results_to_supabase(result)

        # Should not attempt to insert
        mock_client.table.assert_not_called()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_missing_keys(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles missing keys gracefully."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Missing predicted_returns and weights
        result = {
            "date": date(2024, 1, 31),
            "predictions": {"AAPL": 150.25},
            "actual_prices_last_month": {"AAPL": [148.0]},
        }

        save_results_to_supabase(result)

        # Should still work, using defaults
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 1
        assert insert_call_args[0]["ticker"] == "AAPL"
        assert insert_call_args[0]["predicted_price"] == 150.25
        assert insert_call_args[0]["predicted_return"] == 0.0  # Default
        assert insert_call_args[0]["portfolio_weight"] == 0.0  # Default
        assert json.loads(insert_call_args[0]["actual_prices_last_month"]) == [148.0]

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_insert_failure(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles insertion failure."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.side_effect = Exception("Database connection error")
        mock_get_client.return_value = mock_client

        result = {
            "date": date(2024, 1, 31),
            "predictions": {"AAPL": 150.25},
            "predicted_returns": {"AAPL": 0.02},
            "weights": {"AAPL": 1.0},
            "actual_prices_last_month": {"AAPL": [148.0]},
        }

        # Should propagate the exception
        with pytest.raises(Exception, match="Database connection error"):
            save_results_to_supabase(result)
