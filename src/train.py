import os
import json
import logging
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Get config from environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME","iris_model_exp")
MLFLOW_RUN_NAME = os.environ.get("MLFLOW_RUN_NAME","iris-logreg")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME","iris-classifier")
PROMOTION_STAGE = os.environ.get("PROMOTION_STAGE","Staging")

def main():
    """Trains a simple model and logs it to MLflow."""
    logging.info(f"Starting model training for experiment: {MLFLOW_EXPERIMENT_NAME}")
    logging.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )

    C = 1.0
    max_iter = 200

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log the metrics and model
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(clf, name=REGISTERED_MODEL_NAME)
        logging.info(f"Model trained with accuracy: {acc:.4f}")

        # Get the run_id
        run_id = mlflow.active_run().info.run_id
        logging.info(f"MLflow Run ID: {run_id}")
        
        # Print the run_id as JSON to stdout so Airflow can capture it.
        print(json.dumps({"run_id": run_id}))

if __name__ == "__main__":
    main()