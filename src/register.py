"""
Task: Registers a model from a given run_id.
Args:
    run_id (str): The MLflow run_id to register.
Output: Prints a JSON string with 'model_name' and 'model_version' to stdout.
"""
import logging
import os
import sys
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Get config from environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "iris-classifier")


def main(run_id: str):
    """Registers an MLflow model from the specific run ID provided."""
    logging.info(f"Registering model '{REGISTERED_MODEL_NAME}' from run_id: {run_id}")
    logging.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # The artifact path must match what was used in train.py
    model_uri = f"runs:/{run_id}/{REGISTERED_MODEL_NAME}"

    try:
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME
        )
        
        version_str = str(model_version.version)
        logging.info(f"Successfully registered model '{REGISTERED_MODEL_NAME}' version {version_str}")

        # Add a description
        client.update_model_version(
            name=REGISTERED_MODEL_NAME,
            version=version_str,
            description=f"Model version {version_str} registered from run {run_id} via Airflow pipeline."
        )

        # This is the critical output for XCom
        print(json.dumps({
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": version_str
        }))

    except Exception as e:
        logging.error(f"Error registering model: {e}")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help="The MLflow run_id to register.")
    args = parser.parse_args()
    
    main(args.run_id)