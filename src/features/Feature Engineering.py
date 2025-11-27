"""
Generate a lightweight feature set from the daily metrics table without
altering the original data.

The script:
- Loads Data/processed/daily_metrics.parquet (or a supplied path)
- Adds only two interpretable features: day_of_week and is_holiday
- Optionally filters by date
- Saves the feature set to Data/processed/features.parquet (and optional CSV)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML features from daily metrics.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("Data/processed/daily_metrics.parquet"),
        help="Path to the aggregated daily metrics file (Parquet or CSV).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("Data/processed/features.parquet"),
        help="Destination path for the engineered features (Parquet).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("Data/processed/features.csv"),
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Optional ISO date to filter records on/after this date.",
    )
    return parser.parse_args()


def load_daily_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily metrics file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["order_date"])

    if "order_date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "order_date"})

    df["order_date"] = pd.to_datetime(df["order_date"])
    return df.sort_values("order_date").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["order_date"]
    df["day_of_week"] = dt.dt.dayofweek
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=dt.min(), end=dt.max())
    df["is_holiday"] = df["order_date"].isin(holidays).astype(int)
    return df


def build_feature_frame(df: pd.DataFrame, min_date: Optional[str]) -> pd.DataFrame:
    df = add_calendar_features(df)
    if min_date:
        df = df[df["order_date"] >= pd.to_datetime(min_date)]
    return df.reset_index(drop=True)


def save_outputs(df: pd.DataFrame, parquet_path: Path, csv_path: Optional[Path]) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    daily_df = load_daily_metrics(args.input_path)
    feature_df = build_feature_frame(daily_df, args.min_date)
    save_outputs(feature_df, args.output_path, args.csv_path)
    print(
        f"Saved features with {len(feature_df)} rows "
        f"to {args.output_path} (csv: {args.csv_path})."
    )


if __name__ == "__main__":
    main()

# AI usage: This feature builder was written manually without AI support.

