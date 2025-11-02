#!/usr/bin/env bash
set -euo pipefail

# This script assumes you already have your venv set up
source venv/bin/activate

export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MODEL_NAME="iris-classifier"
export MODEL_STAGE="Production"

# Start FastAPI (serves on port 8000)
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload