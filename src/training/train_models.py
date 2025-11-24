"""
Train and evaluate baseline models (Linear Regression, Random Forest) for the
daily order forecasting task. Configurable via configs/train_config.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline forecasting models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to the training configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_split(path: Path, date_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[date_column])
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def prepare_xy(df: pd.DataFrame, targets: List[str], drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = feature_df.drop(columns=targets)
    y = df[targets]
    return X, y


def build_model(name: str, spec: Dict) -> MultiOutputRegressor:
    model_type = spec.get("type")
    if model_type == "linear":
        base_model = LinearRegression()
    elif model_type == "random_forest":
        params = spec.get("params", {})
        base_model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MultiOutputRegressor(base_model)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict:
    metrics = {}
    mae_list = []
    rmse_list = []
    for idx, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
        rmse = mse ** 0.5
        metrics[name] = {"mae": float(mae), "rmse": float(rmse)}
        mae_list.append(mae)
        rmse_list.append(rmse)
    metrics["overall_mae"] = float(np.mean(mae_list))
    metrics["overall_rmse"] = float(np.mean(rmse_list))
    return metrics


def save_artifacts(model, metrics: Dict, model_name: str, artifact_cfg: Dict) -> Tuple[Path, Path]:
    model_dir = Path(artifact_cfg["model_dir"])
    metrics_dir = Path(artifact_cfg["metrics_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}.joblib"
    metrics_path = metrics_dir / f"{model_name}_metrics.json"

    joblib.dump(model, model_path)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return model_path, metrics_path


def save_feature_schema(feature_names: List[str], targets: List[str], artifact_cfg: Dict) -> Path:
    schema_path = Path(artifact_cfg.get("schema_path", Path(artifact_cfg["model_dir"]) / "feature_schema.json"))
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": feature_names,
        "target_columns": targets,
    }
    with schema_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return schema_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    feature_cfg = config.get("features", {})
    artifact_cfg = config["artifacts"]
    targets = data_cfg["target_columns"]
    drop_cols = feature_cfg.get("drop_columns", [])

    train_df = load_split(Path(data_cfg["train_path"]), data_cfg["date_column"])
    val_df = load_split(Path(data_cfg["val_path"]), data_cfg["date_column"])
    test_df = load_split(Path(data_cfg["test_path"]), data_cfg["date_column"])

    X_train, y_train = prepare_xy(train_df, targets, drop_cols)
    X_val, y_val = prepare_xy(val_df, targets, drop_cols)
    X_test, y_test = prepare_xy(test_df, targets, drop_cols)

    feature_names = X_train.columns.tolist()

    for model_name, spec in config["models"].items():
        print(f"\n=== Training {model_name} ===")
        model = build_model(model_name, spec)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics = evaluate_predictions(y_val.values, val_preds, targets)
        test_metrics = evaluate_predictions(y_test.values, test_preds, targets)

        combined_metrics = {
            "validation": val_metrics,
            "test": test_metrics,
        }
        for split_name, metrics in combined_metrics.items():
            print(
                f"{model_name} [{split_name}] -> overall MAE: {metrics['overall_mae']:.2f}, "
                f"overall RMSE: {metrics['overall_rmse']:.2f}"
            )

        save_artifacts(model, combined_metrics, model_name, artifact_cfg)
        save_feature_schema(feature_names, targets, artifact_cfg)


if __name__ == "__main__":
    main()

