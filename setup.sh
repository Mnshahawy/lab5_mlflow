#!/usr/bin/env bash
set -euo pipefail

# Choose your Airflow version here (works well on Py3.10–3.12)
AIRFLOW_VERSION="2.10.2"

# Detect python version (major.minor)
PY_VER=$(python3.11 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')

# Airflow constraints pin
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PY_VER}.txt"

# Create venv
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel setuptools

# Install Airflow with constraints in the SAME venv
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Install the rest
pip install -r requirements.txt
