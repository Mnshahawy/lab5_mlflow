"""
Task: Promotes a model version to a target stage *if* it's better
      than the current model in that stage, or if it's the first.
Args:
    run_id (str): The run_id of the new model version.
    model_name (str): The registered model name.
    model_version (str): The model version to promote.
Output: Logs to stdout. No JSON output required.
"""
import logging
import os
import sys
import argparse
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Get config from environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
PROMOTION_STAGE = os.environ.get("PROMOTION_STAGE", "Staging")

def get_accuracy_from_run(client: MlflowClient, run_id: str) -> float:
    """Fetches the 'accuracy' metric from a given run_id."""
    try:
        run_data = client.get_run(run_id).data
        accuracy = run_data.metrics["accuracy"]
        logging.info(f"Run {run_id} has accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        logging.error(f"Could not get accuracy for run {run_id}: {e}")
        sys.exit(1)

def main(run_id: str, model_name: str, model_version: str):
    """
    Promotes a model if it's the first or has better accuracy
    than the current model in the target stage.
    """
    logging.info(f"Checking promotion criteria for model '{model_name}' v{model_version} (from run {run_id}).")
    logging.info(f"Target stage: '{PROMOTION_STAGE}'")
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # 1. Get new model's accuracy
    new_accuracy = get_accuracy_from_run(client, run_id)
    
    # 2. Find the current model(s) in the target stage
    current_prod_models = client.get_latest_versions(model_name, stages=[PROMOTION_STAGE])
    
    should_promote = False
    if not current_prod_models:
        # 3. If no model is in production, promote this one (first version)
        logging.info(f"No model currently in '{PROMOTION_STAGE}'. Promoting as first version.")
        should_promote = True
    else:
        # 4. If a model exists, compare accuracy
        # Note: get_latest_versions can return multiple, but we'll check the most recent one
        current_model_version = current_prod_models[0]
        logging.info(f"Found current model in '{PROMOTION_STAGE}': version {current_model_version.version} (from run {current_model_version.run_id})")
        
        current_accuracy = get_accuracy_from_run(client, current_model_version.run_id)
        
        if new_accuracy > current_accuracy:
            logging.info(f"New model accuracy ({new_accuracy}) is better than current ({current_accuracy}). Promoting.")
            should_promote = True
        else:
            logging.warning(
                f"New model accuracy ({new_accuracy}) is not better than "
                f"current ({current_accuracy}). Skipping promotion."
            )
            should_promote = False

    # 5. Perform promotion if criteria are met
    if should_promote:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=PROMOTION_STAGE,
                archive_existing_versions=(PROMOTION_STAGE == "Production")
            )
            logging.info(
                f"Successfully promoted model '{model_name}' version {model_version} to '{PROMOTION_STAGE}'"
            )
        except Exception as e:
            logging.error(f"Error promoting model: {e}")
            sys.exit(1)
    else:
        # Exit gracefully if we decided not to promote
        logging.info("Promotion check complete. No action taken.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help="The MLflow run_id of the new model.")
    parser.add_argument("model_name", type=str, help="The registered model name.")
    parser.add_argument("model_version", type=str, help="The model version to promote.")
    args = parser.parse_args()
    
    main(args.run_id, args.model_name, args.model_version)

