#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination s3://my-ml-flow-artifacts \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000

