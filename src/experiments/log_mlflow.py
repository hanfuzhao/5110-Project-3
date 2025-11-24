"""
Log trained models and metrics to MLflow as a standalone experiment-tracking step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log training artifacts to MLflow.")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to the training configuration (for data/model metadata).",
    )
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=Path("configs/mlflow_config.yaml"),
        help="Path to the MLflow configuration.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open() as f:
        return json.load(f)


def flatten_metrics(metrics: Dict) -> Dict[str, float]:
    flat = {}
    for split_name, split_values in metrics.items():
        for metric_key, value in split_values.items():
            if isinstance(value, dict):
                for stat_name, stat_value in value.items():
                    flat[f"{split_name}.{metric_key}.{stat_name}"] = float(stat_value)
            else:
                flat[f"{split_name}.{metric_key}"] = float(value)
    return flat


def read_length(path: Path, date_column: str) -> int:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[date_column])
    return len(df)


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.train_config)
    tracking_cfg = load_yaml(args.tracking_config)

    data_cfg = train_cfg["data"]
    artifact_cfg = train_cfg["artifacts"]
    feature_cfg = train_cfg.get("features", {})
    drop_cols = feature_cfg.get("drop_columns", [])

    mlflow.set_tracking_uri(tracking_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(tracking_cfg.get("experiment_name", "default"))

    lengths = {
        "train": read_length(Path(data_cfg["train_path"]), data_cfg["date_column"]),
        "val": read_length(Path(data_cfg["val_path"]), data_cfg["date_column"]),
        "test": read_length(Path(data_cfg["test_path"]), data_cfg["date_column"]),
    }

    schema_path = Path(artifact_cfg["schema_path"])
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")

    for model_name, spec in train_cfg["models"].items():
        model_path = Path(artifact_cfg["model_dir"]) / f"{model_name}.joblib"
        metrics_path = Path(artifact_cfg["metrics_dir"]) / f"{model_name}_metrics.json"

        metrics = load_metrics(metrics_path)
        flat_metrics = flatten_metrics(metrics)

        params = {
            "model_name": model_name,
            "model_type": spec.get("type"),
            "drop_columns": ",".join(drop_cols),
            "train_rows": lengths["train"],
            "val_rows": lengths["val"],
            "test_rows": lengths["test"],
        }
        extra_params = spec.get("params", {})
        params.update({f"param_{k}": v for k, v in extra_params.items()})

        run_name_prefix = tracking_cfg.get("run_name_prefix", "run")
        with mlflow.start_run(run_name=f"{run_name_prefix}-{model_name}"):
            mlflow.log_params(params)
            for metric_name, value in flat_metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(schema_path)


if __name__ == "__main__":
    main()

