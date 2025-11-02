set -euo pipefail

# Use the repo-local Airflow home
export AIRFLOW_HOME="$(pwd)/.airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False

source venv/bin/activate

# Initialize metadata DB and create an admin user
airflow db init || true
# idempotent user creation
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com || true

echo "Airflow initialized. Home: $AIRFLOW_HOME"