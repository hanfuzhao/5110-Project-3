#!/bin/bash
set -euo pipefail

MODE=${MODE:-serve}

echo "Starting container in mode: $MODE"

if [ "$MODE" = "train" ]; then
  python src/training/train_models.py --config "${TRAIN_CONFIG:-configs/train_config.yaml}"
elif [ "$MODE" = "serve" ]; then
  exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8080}"
else
  echo "Unknown MODE: $MODE"
  exit 1
fi

