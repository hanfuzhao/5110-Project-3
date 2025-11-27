"""
Build the daily order-metrics table used by the forecasting pipeline.

- Read orders.csv (or any file with the same schema)
- Aggregate order volume, GMV, discounts, taxes, etc. by day
- Persist the result to Data/processed/daily_metrics.parquet (default)

Downstream training/inference scripts can consume this table directly to
ensure reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw orders into daily metrics.")
    parser.add_argument(
        "--orders-path",
        type=Path,
        default=Path("Data/orders.csv"),
        help="Path to the raw orders CSV file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("Data/processed/daily_metrics.parquet"),
        help="Output path for the aggregated Parquet file.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV output path; leave empty to skip CSV export.",
    )
    return parser.parse_args()


def load_orders(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Orders file not found: {path}")
    df = pd.read_csv(path, parse_dates=["order_datetime"])
    return df


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "order_date"
    df[date_col] = df["order_datetime"].dt.date

    base_agg = df.groupby(date_col).agg(
        order_count=("order_id", "nunique"),
        item_count=("num_items", "sum"),
        subtotal=("subtotal", "sum"),
        shipping_fee=("shipping_fee", "sum"),
        tax=("tax", "sum"),
        discount_total=("discount_total", "sum"),
        revenue=("total", "sum"),
    )

    # Append status distribution to analyze cancellation/return trends later.
    status = (
        df.pivot_table(
            index=date_col,
            columns="order_status",
            values="order_id",
            aggfunc="count",
            fill_value=0,
        )
        .astype(int)
        .add_prefix("status_")
    )

    daily = base_agg.join(status).sort_index()
    daily["avg_order_value"] = daily["revenue"] / daily["order_count"].replace(0, pd.NA)

    # Convert index to DatetimeIndex for downstream time-series models.
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "order_date"

    return daily


def save_outputs(df: pd.DataFrame, parquet_path: Path, csv_path: Optional[Path]) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path)


def main() -> None:
    args = parse_args()
    orders_df = load_orders(args.orders_path)
    daily_df = build_daily_metrics(orders_df)
    save_outputs(daily_df, args.output_path, args.csv_path)
    print(
        f"Saved daily metrics with {len(daily_df)} rows "
        f"to {args.output_path} (and CSV: {args.csv_path})."
    )


if __name__ == "__main__":
    main()

# AI usage: This script was authored without relying on any AI assistance.
