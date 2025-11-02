# Prophet Forecasting for Portfolio Optimisation

An end-to-end machine learning project that forecasts stock and asset prices using LightGBM (Gradient Boosting) models, then applies Markowitz portfolio optimization to rebalance portfolios based on these forecasts.

## Project Overview

This project combines machine learning-based time series forecasting with modern portfolio theory to create an automated trading and portfolio management system. The workflow consists of two main components:

1. **LightGBM Price Forecasting**: Uses gradient boosting to predict future asset prices
2. **Markowitz Portfolio Optimization**: Optimizes portfolio allocation based on forecasted returns and risk

## Components

### 1. LightGBM (Gradient Boosting Machine)

**What is LightGBM?**

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It's designed to be distributed and efficient, with faster training speed and lower memory usage compared to traditional gradient boosting methods.

**Why LightGBM for Stock Prediction?**

LightGBM is excellent for tabular time series data because it:
- Handles non-linear patterns in price movements effectively
- Works well with lagged features and historical data
- Trains faster than deep learning models like LSTM
- Provides feature importance insights
- Requires minimal feature engineering compared to neural networks

**How It Works in This Project:**

- Input: Historical price data with lagged prices, moving averages, and returns
- Architecture: Gradient boosting trees that learn patterns from historical features
- Output: Forecasted prices for each asset in the portfolio for the next time period
- Training: The model learns from historical data to minimize prediction error

### 2. Markowitz Portfolio Optimization

**What is Markowitz Portfolio Optimization?**

Markowitz portfolio optimization, also known as Modern Portfolio Theory (MPT), is a mathematical framework for constructing optimal portfolios. Developed by Harry Markowitz in 1952, it balances the trade-off between expected returns and risk.

**Key Concepts:**

- **Expected Return**: The weighted average of expected returns of individual assets
- **Risk (Volatility)**: Measured as the standard deviation of portfolio returns
- **Correlation**: How assets move relative to each other
- **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

**The Optimization Problem:**

The Markowitz model solves:

```
Maximize: μᵀw - λ(wᵀΣw)

Subject to:
- Σwᵢ = 1 (weights sum to 1)
- wᵢ ≥ 0 (long-only portfolio, optional)
- Additional constraints (sector limits, etc.)
```

Where:
- `μ` = vector of expected returns (from LightGBM price forecasts)
- `Σ` = covariance matrix of asset returns
- `w` = portfolio weights
- `λ` = risk aversion parameter

**How It Works in This Project:**

1. **Input**: Forecasted returns (derived from LightGBM price predictions) for each asset
2. **Risk Estimation**: Historical covariance matrix calculated from asset returns
3. **Optimization**: Solves for optimal weights that maximize risk-adjusted returns
4. **Output**: Recommended portfolio allocation (weights for each asset)
5. **Rebalancing**: Portfolio is rebalanced based on these optimal weights

## Project Workflow

```
Historical Data
    ↓
Feature Engineering
    ↓
LightGBM Training
    ↓
Price Forecasting
    ↓
Return Calculation (from prices)
    ↓
Expected Returns & Risk Metrics
    ↓
Markowitz Optimization
    ↓
Optimal Portfolio Weights
    ↓
Portfolio Rebalancing
```

## Key Features

- **Machine Learning Forecasting**: LightGBM gradient boosting for price prediction
- **Risk-Aware Optimization**: Incorporates covariance and correlation for risk management
- **Automated Rebalancing**: Systematic portfolio adjustments based on forecasts
- **Backtesting Framework**: Evaluate strategy performance on historical data
- **Multi-Asset Support**: Handle portfolios with multiple stocks/assets simultaneously

## Project Structure

```
Machine-Learning-For-Portfolio-Optimisation/
├── README.md
├── pyproject.toml
├── Makefile
├── src/
│   ├── data/
│   ├── models/
│   ├── optimization/
│   └── utils/
├── notebooks/
├── tests/
└── data/
```

## Installation

### Standard Installation

```bash
# Install dependencies using Poetry
make install

# Or manually
poetry install
```

### Installation on M-chip Mac (Apple Silicon)

If you're using an M-chip Mac (Apple Silicon), you'll need to install additional dependencies for LightGBM:

```bash
# Install Homebrew if you don't have it
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv and libomp
brew install pyenv
brew install libomp

# Then proceed with standard installation
make install
# or
poetry install
```

**Note**: `libomp` (OpenMP library) is required for LightGBM to compile correctly on Apple Silicon Macs.

## Usage

*(To be added as the project develops)*

## Dependencies

- **Machine Learning**: LightGBM for gradient boosting price prediction
- **Data Processing**: Pandas, NumPy for data manipulation
- **Optimization**: SciPy for portfolio optimization solvers
- **Data Sources**: yfinance or similar APIs for stock data
- **Visualization**: Matplotlib, Seaborn for results visualization

## License

*(To be determined)*

## CircleCI

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/XbC7AoPbDq6kv77Q92i6dK/9FPwBXXN1scw4EnPpD9g5m/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/XbC7AoPbDq6kv77Q92i6dK/9FPwBXXN1scw4EnPpD9g5m/tree/main)

## References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

