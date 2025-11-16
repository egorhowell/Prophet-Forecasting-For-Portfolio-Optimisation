"""Tests for portfolio optimisation module."""

import numpy as np
import pandas as pd

from src.optimiser import (
    calculate_mean_variance,
    optimize_portfolio_mean_variance,
)


class TestPortfolioOptimisation:
    """Test portfolio optimisation functions."""

    def test_calculate_mean_variance(self) -> None:
        """Test calculating mean and covariance from Returns columns."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create DataFrames with Returns column (as the function expects)
        df1 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 100,
                "Returns": np.random.randn(100) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 50,
                "Returns": np.random.randn(100) * 0.02,
            },
            index=[d.date() for d in dates],
        )

        data_dict = {"ASSET1": df1, "ASSET2": df2}
        mean_returns, cov_matrix = calculate_mean_variance(data_dict)

        assert isinstance(mean_returns, pd.Series)
        assert isinstance(cov_matrix, pd.DataFrame)
        assert len(mean_returns) == 2
        assert cov_matrix.shape == (2, 2)
        assert "ASSET1" in mean_returns.index
        assert "ASSET2" in mean_returns.index

    def test_optimize_portfolio_mean_variance_basic(self) -> None:
        """Test basic portfolio optimisation."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        df1 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 100,
                "Returns": np.random.randn(100) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 50,
                "Returns": np.random.randn(100) * 0.02,
            },
            index=[d.date() for d in dates],
        )

        data_dict = {"ASSET1": df1, "ASSET2": df2}
        optimal_weights = optimize_portfolio_mean_variance(data_dict)

        assert isinstance(optimal_weights, dict)
        assert len(optimal_weights) == 2
        assert np.isclose(sum(optimal_weights.values()), 1.0, rtol=1e-5)
        assert all(w >= 0 and w <= 1 for w in optimal_weights.values())
        assert "ASSET1" in optimal_weights
        assert "ASSET2" in optimal_weights

    def test_optimize_portfolio_mean_variance_with_minimum_allocation(self) -> None:
        """Test portfolio optimisation with custom minimum allocation."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        df1 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 100,
                "Returns": np.random.randn(100) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": np.random.randn(100) * 10 + 50,
                "Returns": np.random.randn(100) * 0.02,
            },
            index=[d.date() for d in dates],
        )

        data_dict = {"ASSET1": df1, "ASSET2": df2}
        min_allocation = 0.1  # 10% minimum

        optimal_weights = optimize_portfolio_mean_variance(
            data_dict, minimum_allocation=min_allocation
        )

        assert isinstance(optimal_weights, dict)
        assert len(optimal_weights) == 2
        assert np.isclose(sum(optimal_weights.values()), 1.0, rtol=1e-5)
        assert all(w >= min_allocation for w in optimal_weights.values())
        assert all(w <= 1.0 for w in optimal_weights.values())
