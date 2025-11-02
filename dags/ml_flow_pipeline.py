from datetime import datetime, timedelta
import os, logging, json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import sys, subprocess, pathlib, shlex

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
ENV_VARS = {
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
    "MLFLOW_EXPERIMENT_NAME": "iris_model_exp",
    "MLFLOW_RUN_NAME": "iris-logreg",
    "REGISTERED_MODEL_NAME": "iris-classifier",
    "PROMOTION_STAGE": "Staging",
    "PYTHONPATH": REPO_ROOT + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")
}

default_args = {
    "owner": "you",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

def _run_script(script_name: str, *args, **context) -> dict:
    """
    Helper function to run an external Python script and handle I/O.
    
    The script is expected to print its final output as a single
    JSON line to stdout.
    """
    script_path = str(pathlib.Path(REPO_ROOT) / "src" / script_name)
    cmd = [sys.executable, script_path] + list(args)

    logging.info(f"[Runner] Executing: {' '.join(shlex.quote(c) for c in cmd)}")
    logging.info(f"[Runner] CWD: {REPO_ROOT}")

    # Set up environment
    env = os.environ.copy()
    env.update(ENV_VARS)
     
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.stdout:
        logging.info("Script STDOUT:\n%s", proc.stdout)
    if proc.stderr:
        logging.error("Script STDERR:\n%s", proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {proc.returncode}")

    # Parse the last line of stdout as JSON
    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        return json.loads(last_line)
    except (IndexError, json.JSONDecodeError):
        logging.warning(f"Script {script_name} did not produce valid JSON output.")
        return {}

def run_training(**context) -> dict:
    """
    Wrapper for the 'train.py' script.
    Pushes the 'run_id' to XCom.
    """
    logging.info("Starting model training script...")
    output_json = _run_script("train.py", **context)
    if "run_id" not in output_json:
        raise AirflowException("train.py did not return a 'run_id' in its JSON output.")
    
    logging.info(f"Training script finished. Run ID: {output_json['run_id']}")
    # Airflow pushes this dict to XCom(Cross Communication between tasks)
    return output_json


def run_registration(**context):
    """
    Wrapper for the 'register.py' script.
    Pulls the 'run_id' from XCom and passes it as an arg.
    Pushes the 'model_name' and 'model_version' to XCom.
    """
    logging.info("Starting model registration script...")

    # Pull the output from the 'train_model' task
    ti = context['task_instance']
    train_output = ti.xcom_pull(task_ids='train_model')
    run_id = train_output['run_id']
    
    logging.info(f"Registering model from run_id: {run_id}")
    output_json = _run_script("register.py", run_id, **context) # Pass run_id as arg
    
    if "model_name" not in output_json or "model_version" not in output_json:
        raise AirflowException("register.py did not return 'model_name' or 'model_version'.")

    logging.info(f"Registration script finished. Model: {output_json['model_name']}, Version: {output_json['model_version']}")
    return output_json


def run_promotion(**context):
    """
    Wrapper for the 'promote.py' script.
    Pulls 'run_id', 'model_name' and 'model_version' from XCom and passes them as args.
    """
    logging.info("Starting model promotion script...")
    # Pull the output from the 'register_model' task
    ti = context['task_instance']
    train_output = ti.xcom_pull(task_ids='train_model')
    run_id = train_output['run_id']
    register_output = ti.xcom_pull(task_ids='register_model')
    model_name = register_output['model_name']
    model_version = register_output['model_version']
    
    logging.info(f"Promoting model: {model_name} version {model_version} (from run {run_id})")
    _run_script("promote.py", run_id, model_name, model_version, **context)
    logging.info("Promotion script finished.")

with DAG(
    dag_id="ml_flow_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual only
    catchup=False,
    default_args=default_args,
    description="Train a tiny model and log to MLflow",
) as dag:
    # Define the tasks
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=run_registration,
    )
    
    promote_model_task = PythonOperator(
        task_id="promote_model",
        python_callable=run_promotion,
    )

    # Define the sequential dependency
    train_model_task >> register_model_task >> promote_model_task