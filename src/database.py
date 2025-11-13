"""Database operations for saving optimisation results to Supabase."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from supabase import Client, create_client

from src.settings import SUPABASE_TABLE_NAME

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client | None:
    """
    Create and return Supabase client from environment variables.

    Returns:
        Supabase client if credentials are available, None otherwise
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.warning("Supabase credentials not found in environment variables")
        return None

    return create_client(url, key)


def save_results_to_supabase(result: dict[str, Any]) -> None:
    """
    Save optimisation results to Supabase database.

    Args:
        result: Dictionary containing optimisation results from run_optimisation()
            Expected keys: predictions, predicted_returns, weights

    Raises:
        ValueError: If Supabase client cannot be created or insertion fails
    """
    supabase =\
        get_supabase_client()
    if supabase is None:
        raise ValueError(
            "Supabase client not available. Check SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    as_of_date = result.get("date")
    predictions = result.get("predictions", {})
    predicted_returns = result.get("predicted_returns", {})
    weights = result.get("weights", {})
    actual_prices_last_month = result.get("actual_prices_last_month", {})

    if not predictions:
        logger.warning("No predictions to save")
        return

    # Prepare rows for insertion - one row per stock
    rows = []
    for ticker in predictions.keys():
        row = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "ticker": ticker,
            "predicted_price": float(predictions.get(ticker, 0.0)),
            "predicted_return": float(predicted_returns.get(ticker, 0.0)),
            "actual_prices_last_month": json.dumps(actual_prices_last_month.get(ticker, [])),
            "portfolio_weight": float(weights.get(ticker, 0.0)),
        }
        rows.append(row)

    logger.info(f"Inserting {len(rows)} rows into Supabase...")
    (supabase.table(SUPABASE_TABLE_NAME).insert(rows).execute())

    logger.info(f"Successfully saved {len(rows)} predictions to Supabase")
