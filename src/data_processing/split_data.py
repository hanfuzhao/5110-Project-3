"""
Create reproducible train/validation/test splits for the forecasting task.

The script reads the processed daily dataset (defaults to the lightweight
feature file), orders it chronologically, and slices it into three contiguous
time windows to avoid data leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split time-series data into train/val/test.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("Data/processed/features.parquet"),
        help="Processed dataset path (Parquet or CSV).",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default="order_date",
        help="Column containing the chronological ordering.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2024-12-31",
        help="Inclusive date marking the end of the training window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2025-09-30",
        help="Inclusive date marking the end of the validation window; remaining rows form the test set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/processed/splits"),
        help="Directory to store the split datasets.",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["order_date"])
    if "order_date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "order_date"})
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df.sort_values("order_date").reset_index(drop=True)


def split_by_date(
    df: pd.DataFrame, date_col: str, train_end: str, val_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be earlier than val_end.")

    train_mask = df[date_col] <= train_end_ts
    val_mask = (df[date_col] > train_end_ts) & (df[date_col] <= val_end_ts)
    test_mask = df[date_col] > val_end_ts

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    for split_name, split_df in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        if split_df.empty:
            raise ValueError(f"{split_name} split is empty; adjust the date boundaries.")

    return train_df, val_df, test_df


def save_split(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()
    dataset = load_dataframe(args.input_path)
    train_df, val_df, test_df = split_by_date(dataset, args.date_column, args.train_end, args.val_end)

    save_split(train_df, "train", args.output_dir)
    save_split(val_df, "val", args.output_dir)
    save_split(test_df, "test", args.output_dir)

    print(
        "Splits saved to "
        f"{args.output_dir} "
        f"(train={len(train_df)} rows, val={len(val_df)}, test={len(test_df)})."
    )


if __name__ == "__main__":
    main()

# AI usage: ChatGPT was referenced when outlining the initial CLI structure.

