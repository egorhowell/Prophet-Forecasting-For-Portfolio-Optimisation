"""Streamlit dashboard for Prophet-based portfolio forecasts."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

st.set_page_config(page_title="Portfolio Forecast Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_supabase_predictions() -> pd.DataFrame:
    """Return latest Supabase rows (one per ticker per date)."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    response = (
        client.table(SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    data = getattr(response, "data", None)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    df = df.sort_values(["as_of_date", "created_at"], ascending=[True, False])
    df = df.drop_duplicates(subset=["as_of_date", "ticker"], keep="first")

    if "actual_prices_last_month" in df.columns:
        df["actual_prices_last_month"] = df["actual_prices_last_month"].apply(_parse_price_history)

    return df


def _parse_price_history(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [float(value) for value in decoded]
    return []


def _latest_actual_price(row: pd.Series) -> float | None:
    prices = row.get("actual_prices_last_month", [])
    if prices:
        return float(prices[-1])
    return None


def build_price_history(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    prices = row.get("actual_prices_last_month", [])
    if not prices:
        return None

    as_of_date: date = row["as_of_date"]
    n = len(prices)

    actual_index = pd.bdate_range(end=pd.to_datetime(as_of_date), periods=n)
    actual_df = pd.DataFrame({"date": actual_index, "price": prices})

    prediction_date = pd.bdate_range(
        start=pd.to_datetime(as_of_date) + pd.Timedelta(days=1),
        periods=1,
    )[0]
    predicted_df = pd.DataFrame({"date": [prediction_date], "price": [row["predicted_price"]]})

    return actual_df, predicted_df


@lru_cache(maxsize=1)
def compute_prediction_performance(data_json: str) -> pd.DataFrame:
    """Compare past predictions against actual outcomes using successive days."""
    df = pd.read_json(data_json, orient="records", convert_dates=False)
    if df.empty:
        return df

    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "actual_prices_last_month" in df.columns:
        df["actual_prices_last_month"] = df["actual_prices_last_month"].apply(_parse_price_history)
    df = df.sort_values(["ticker", "as_of_date"])

    records: list[dict[str, object]] = []

    for ticker, group in df.groupby("ticker"):
        group = group.reset_index(drop=True)
        for idx in range(len(group) - 1):
            current = group.loc[idx]

            prices = current.get("actual_prices_last_month")
            if not prices:
                continue

            next_row = group.loc[idx + 1]
            actual_next_price = _latest_price_from_row(next_row)
            if actual_next_price is None:
                continue

            records.append(
                {
                    "ticker": ticker,
                    "prediction_date": current["as_of_date"],
                    "evaluation_date": next_row["as_of_date"],
                    "predicted_price": float(current["predicted_price"]),
                    "actual_price": actual_next_price,
                    "error": actual_next_price - float(current["predicted_price"]),
                }
            )

    perf_df = pd.DataFrame(records)
    if perf_df.empty:
        return perf_df

    perf_df["absolute_error"] = perf_df["error"].abs()
    perf_df["error_pct"] = perf_df["error"] / perf_df["predicted_price"]
    return perf_df


def _latest_price_from_row(row: pd.Series) -> float | None:
    prices = row.get("actual_prices_last_month")
    if isinstance(prices, list) and prices:
        return float(prices[-1])
    return None


def pie_chart(weights_df: pd.DataFrame):
    chart_df = weights_df[["ticker", "portfolio_weight"]].copy()
    chart_df["portfolio_weight"] = pd.to_numeric(chart_df["portfolio_weight"], errors="coerce")
    chart_df = chart_df.dropna(subset=["portfolio_weight"])

    total_weight = chart_df["portfolio_weight"].sum()
    if total_weight <= 0:
        return None

    fig = px.pie(
        chart_df,
        names="ticker",
        values="portfolio_weight",
        hole=0.3,
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:.2f}")
    fig.update_layout(showlegend=True, legend_title_text="Ticker", height=360)
    return fig


def run_dashboard() -> None:
    st.title("📊 Portfolio Forecast Dashboard")
    st.caption("Latest Prophet predictions, portfolio weights, and performance analysis sourced from Supabase.")

    df = load_supabase_predictions()
    if df.empty:
        st.info("No prediction data available. Run the optimisation pipeline to populate Supabase.")
        return

    available_dates = sorted(df["as_of_date"].unique(), reverse=True)
    selected_date = st.selectbox("Select as-of date", options=available_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))

    date_df = df[df["as_of_date"] == selected_date].copy().sort_values("ticker")

    st.subheader("Portfolio Weights")
    weight_col, table_col = st.columns([1, 1])
    with weight_col:
        pie = pie_chart(date_df)
        if pie is None:
            st.info("Weights are zero or missing for this date.")
        else:
            st.plotly_chart(pie, use_container_width=True)

    with table_col:
        summary_table = date_df[["ticker", "predicted_price", "predicted_return"]].copy()
        summary_table["predicted_return_pct"] = summary_table["predicted_return"] * 100
        summary_table = summary_table.rename(
            columns={
                "ticker": "Ticker",
                "predicted_price": "Predicted Price",
                "predicted_return_pct": "Predicted Return (%)",
            }
        )
        st.dataframe(
            summary_table[["Ticker", "Predicted Price", "Predicted Return (%)"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted Price": st.column_config.NumberColumn(format="$%.2f"),
                "Predicted Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    tickers = date_df["ticker"].tolist()
    selected_ticker = st.selectbox("Select ticker for detail view", options=tickers, index=0)

    ticker_row = date_df.set_index("ticker").loc[selected_ticker]
    latest_actual = _latest_actual_price(ticker_row)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest Actual Price", f"${latest_actual:.2f}" if latest_actual is not None else "—")
    with col2:
        st.metric("Predicted Price", f"${ticker_row['predicted_price']:.2f}")
    with col3:
        st.metric("Predicted Return", f"{ticker_row['predicted_return']*100:.2f}%")

    st.subheader(f"Price Trend · {selected_ticker}")
    price_history = build_price_history(ticker_row)
    if price_history is not None:
        actual_df, predicted_df = price_history
        actual_chart = (
            alt.Chart(actual_df)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("price:Q", title="Price (USD)"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="Actual Price", format=".2f"),
                ],
            )
        )
        predicted_chart = (
            alt.Chart(predicted_df)
            .mark_point(size=120, color="#ff7f0e")
            .encode(
                x="date:T",
                y="price:Q",
                tooltip=[
                    alt.Tooltip("date:T", title="Prediction Date"),
                    alt.Tooltip("price:Q", title="Predicted Price", format=".2f"),
                ],
            )
        )
        st.altair_chart(actual_chart + predicted_chart, use_container_width=True)
        st.caption("Blue line shows the last month of actual prices; the orange dot is the predicted next-day price.")
    else:
        st.info("No historical price data available for this ticker.")

    st.subheader("Prediction Accuracy")
    perf_df = compute_prediction_performance(df.to_json(orient="records", date_format="iso"))
    if perf_df.empty:
        st.info("Not enough historical runs to evaluate predictions yet. Check back after multiple runs.")
    else:
        ticker_perf = perf_df[perf_df["ticker"] == selected_ticker].copy()
        if ticker_perf.empty:
            st.info("No historical prediction data for this ticker yet.")
        else:
            ticker_perf["error_pct"] = ticker_perf["error_pct"] * 100
            ticker_perf_display = ticker_perf.rename(
                columns={
                    "prediction_date": "Prediction Date",
                    "evaluation_date": "Evaluation Date",
                    "predicted_price": "Predicted Price",
                    "actual_price": "Actual Price",
                    "error": "Error",
                    "absolute_error": "Absolute Error",
                    "error_pct": "Error (%)",
                }
            )
            st.dataframe(
                ticker_perf_display[
                    [
                        "Prediction Date",
                        "Evaluation Date",
                        "Predicted Price",
                        "Actual Price",
                        "Error",
                        "Absolute Error",
                        "Error (%)",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Predicted Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Actual Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Error": st.column_config.NumberColumn(format="$%.2f"),
                    "Absolute Error": st.column_config.NumberColumn(format="$%.2f"),
                    "Error (%)": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

            error_chart = (
                alt.Chart(ticker_perf)
                .mark_bar()
                .encode(
                    x=alt.X("evaluation_date:T", title="Evaluation Date"),
                    y=alt.Y("error:Q", title="Actual - Predicted ($)"),
                    tooltip=[
                        alt.Tooltip("prediction_date:T", title="Prediction Date"),
                        alt.Tooltip("evaluation_date:T", title="Evaluation Date"),
                        alt.Tooltip("predicted_price:Q", title="Predicted", format=".2f"),
                        alt.Tooltip("actual_price:Q", title="Actual", format=".2f"),
                        alt.Tooltip("error:Q", title="Error", format=".2f"),
                    ],
                    color=alt.condition("datum.error > 0", alt.value("#2ca02c"), alt.value("#d62728")),
                )
            )
            st.altair_chart(error_chart, use_container_width=True)


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()

