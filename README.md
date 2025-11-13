# Prophet Forecasting for Portfolio Optimisation

## Project Overview
An end-to-end machine learning project that forecasts stock and asset prices using Facebook/Meta Prophet time series forecasting model, then applies Markowitz portfolio optimisation to rebalance portfolios based on these forecasts.

_(Needless to say it's for illustrative purposes and not financial advice)._

**Live Application**: This project is hosted on a Hostinger VPS, runs every morning at 9am UTC and is accessible at [portfolio-optimisation.com](https://portfolio-optimisation.com)

## Components

### 1. Prophet (Time Series Forecasting)

**What is Prophet?**

Prophet is Facebook's open-source time series forecasting tool designed for business forecasting. It handles trends, seasonality, and holidays automatically, making it robust and easy to use for forecasting time series data.

**How It Works in This Project:**

- Input: Historical price time series with datetime index
- Model: Prophet fits additive components (trend, seasonality, holidays)
- Output: Forecasted prices for each asset in the portfolio for the next trading day
- Training: The model fits to historical price data and generates one-step-ahead forecasts

### 2. Markowitz Portfolio Optimisation

**What is Markowitz Portfolio Optimisation?**

Markowitz portfolio optimisation, also known as Modern Portfolio Theory (MPT), is a mathematical framework for constructing optimal portfolios. Developed by Harry Markowitz in 1952, it balances the trade-off between expected returns and risk.

**Key Concepts:**

- **Expected Return**: The weighted average of expected returns of individual assets
- **Risk (Volatility)**: Measured as the standard deviation of portfolio returns
- **Correlation**: How assets move relative to each other
- **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

**The Optimisation Problem:**

```
Maximize: μᵀw - λ(wᵀΣw)

Subject to:
- Σwᵢ = 1 (weights sum to 1)
- wᵢ ≥ 0 (long-only portfolio, optional)
- Additional constraints (sector limits, etc.)
```

Where:
- `μ` = vector of expected returns (from Prophet price forecasts)
- `Σ` = covariance matrix of asset returns
- `w` = portfolio weights
- `λ` = risk aversion parameter (configurable in `src/settings.py`)

**How It Works in This Project:**

1. **Input**: Forecasted returns (derived from Prophet price predictions) for each asset
2. **Risk Estimation**: Historical covariance matrix calculated from asset returns
3. **Optimisation**: Solves for optimal weights that maximise risk-adjusted returns using SciPy's SLSQP solver
4. **Output**: Recommended portfolio allocation (weights for each asset)
5. **Rebalancing**: Portfolio is rebalanced based on these optimal weights

## Project Workflow

```
Historical Data Extraction (yfinance)
    ↓
Data Preprocessing & Alignment
    ↓
Prophet Model Training (per ticker)
    ↓
Price Forecasting (next day)
    ↓
Return Calculation (from prices)
    ↓
Mean Returns & Covariance Matrix
    ↓
Markowitz Optimisation (SciPy)
    ↓
Optimal Portfolio Weights
    ↓
Results Logging & Output
```

## Project Structure

```
Prophet-Forecasting-For-Portfolio-Optimisation/
├── README.md
├── pyproject.toml          # Poetry dependencies and project config
├── poetry.lock             # Locked dependency versions
├── Makefile                # Convenience commands (install, test, lint, etc.)
├── .circleci/
│   └── config.yml          # CircleCI CI/CD configuration
├── .github/
│   └── workflows/
│       ├── daily-optimisation.yml  # Daily GitHub Actions workflow (runs at 9am UTC)
│       └── deploy.yml              # Deployment workflow to Hostinger VPS
├── scripts/
│   └── deploy.sh           # Deployment script
├── src/
│   ├── __init__.py
│   ├── main.py             # Main entry point and run_optimisation()
│   ├── data.py             # Data extraction and preprocessing
│   ├── database.py         # Database operations
│   ├── model.py            # ProphetModel class
│   ├── optimiser.py        # Portfolio optimisation functions
│   ├── settings.py         # Configuration constants (tickers, risk params)
│   └── streamlit_app.py    # Streamlit web application
├── tests/
    ├── __init__.py
    ├── test_data.py        # Tests for data module
    ├── test_database.py    # Tests for database module
    ├── test_model.py       # Tests for Prophet model
    └── test_optimiser.py   # Tests for optimisation functions

```

## Installation

### Standard Installation

```bash
# Install dependencies using Poetry
make install-dev

# Or manually
poetry install
```

### Requirements

- Python 3.12+
- Poetry (for dependency management)

The project uses Poetry for dependency management. If you don't have Poetry installed:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

## Usage

### Basic Usage

```bash
poetry run python -m src.main
```

Or using the Makefile:

```bash
make run
```

### Configuration

Edit `src/settings.py` to customise:

- **Portfolio Tickers**: Modify `PORTFOLIO_TICKERS` list
- **Risk Aversion**: Adjust `RISK_AVERSION` (higher = more risk averse)
- **Minimum Allocation**: Change `MINIMUM_ALLOCATION` (minimum weight per asset)
- **Date Range**: Update `START_DATE` and `END_DATE` for historical data

Example:

```python
# src/settings.py
PORTFOLIO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
RISK_AVERSION = 3  # Higher = more risk averse
MINIMUM_ALLOCATION = 0.05  # 5% minimum per asset
START_DATE = "2024-01-01"
```

### Programmatic Usage

```python
from src.main import run_optimisation

result = run_optimisation(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    minimum_allocation=0.1  # 10% minimum
)

print(f"Optimal Weights: {result['weights']}")
print(f"Predicted Returns: {result['predicted_returns']}")
```

