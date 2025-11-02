# app/server.py
import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"
MODEL_STAGE         = "Staging"

# Configure MLFlow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = mlflow.tracking.MlflowClient()

# This global dictionary will hold our application's state, including the loaded model
# and its details.
app_state = {
    "current_model": None,
    # Will store dict with version, timestamps, etc.
    "current_model_details": None
}

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

# Define the request body for switching the model.
class SwitchModelRequest(BaseModel):
    version: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"version": "5"},
                {"version": "Staging"}
            ]
        }
    }


# Helper function to fetch model version details from MLFlow
def get_model_version_details(version_str: str) -> dict:
    """
    Fetches model version details from MLFlow based on a version number or stage.
    """
    if mlflow_client is None:
        raise HTTPException(status_code=503, detail="MLFlow client not initialized.")
        
    try:
        model_version_details = None
        if version_str.isdigit():
            # It's a specific version number
            model_version_details = mlflow_client.get_model_version(MODEL_NAME, version_str)
        else:
            # It's a stage (e.g., "Stable", "Staging")
            latest_versions = mlflow_client.get_latest_versions(MODEL_NAME, stages=[version_str])
            if latest_versions:
                model_version_details = latest_versions[0]
        
        if model_version_details:
            # Convert timestamps to human-readable format
            created = datetime.fromtimestamp(model_version_details.creation_timestamp / 1000).isoformat()
            last_promoted = datetime.fromtimestamp(model_version_details.last_updated_timestamp / 1000).isoformat()
            
            return {
                "model_name": model_version_details.name,
                "version": model_version_details.version,
                "stage": model_version_details.current_stage,
                "run_id": model_version_details.run_id,
                "created_timestamp": created,
                "last_promoted_timestamp": last_promoted,
                "description": model_version_details.description
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No model version found for '{MODEL_NAME}' with version/stage '{version_str}'."
            )
            
    except MlflowException as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Failed to find model '{MODEL_NAME}' version/stage '{version_str}'. Details: {e}"
        )

# Define the lifespan of the application to load the model and its details on startup and unload it on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print(f"Attempting to load '{MODEL_STAGE}' version of model: '{MODEL_NAME}'")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    try:
        # Load the latest stable model version from MLFlow
        app_state["current_model"] = mlflow.pyfunc.load_model(model_uri)
        # Get and store its details
        app_state["current_model_details"] = get_model_version_details(MODEL_STAGE)
        
        print(f"Successfully loaded model '{MODEL_NAME}' version '{MODEL_STAGE}'.")
        print(f"Model details: {app_state['current_model_details']}")
            
    except (MlflowException, HTTPException) as e:
        # This error occurs if the model, stage, or MLFlow server is not found
        print(f"Error: Could not load 'Stable' model from MLFlow. No model is active.")
        print(f"Details: {e}")
    except Exception as e:
        # Catch any other potential errors during loading
        print(f"An unexpected error occurred during model loading: {e}")

    # --- API is now running ---
    yield
    # --- Shutdown Logic ---
    print("Shutting down API...")
    app_state["current_model"] = None
    app_state["current_model_details"] = None
    print("Model unloaded.")

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["health"])
def health():
    # This will show the default model URI that is being served.
    return {"status": "ok", "model_uri": f"models:/{MODEL_NAME}/{MODEL_STAGE}"}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Perform prediction on a list of Iris samples using the currently active model.
    """
    if app_state["current_model"] is None:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="No model is currently loaded. Please load a model via startup or the /switch_model_version endpoint."
        )
    
    try:
        # Convert the list of IrisSample objects into a DataFrame
        # Pydantic's .model_dump() is preferred over .dict() in v2
        input_data = [sample.model_dump() for sample in req.samples]
        input_df = pd.DataFrame(input_data)
        
        # Ensure column order matches the model's expectation if it's strict
        # For Iris, this is generally: sepal_length, sepal_width, petal_length, petal_width
        # Re-ordering just in case
        input_df = input_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        
        # Perform prediction
        raw_predictions = app_state["current_model"].predict(input_df)
        
        # Convert predictions (likely numpy array) to a list of ints
        class_ids = [int(pred) for pred in raw_predictions]
        
        # Map class IDs to human-readable labels
        class_labels = [IRIS_LABELS.get(id, "unknown") for id in class_ids]
        
        return PredictResponse(class_id=class_ids, class_label=class_labels)
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during prediction: {e}"
        )

# GET for model details
@app.get("/model", tags=["model"])
def get_model_details() -> dict:
    """
    Get the details of the currently serving model.
    """
    return app_state["current_model_details"]

# POST for switching the model
@app.post("/model", tags=["model"])
def switch_model(req: SwitchModelRequest) -> dict:
    """
    Switch the model to a new version.
    """
    version_str = req.version
    print(f"Attempting to switch to model '{MODEL_NAME}' version/stage: '{version_str}'")
    model_uri = f"models:/{MODEL_NAME}/{version_str}"

    try:
        # This line validates the version exists by trying to load it
        new_model = mlflow.pyfunc.load_model(model_uri)
        # Get and store its details
        new_model_details = get_model_version_details(version_str)
        
        # If successful, update the global state
        app_state["current_model"] = new_model
        app_state["current_model_details"] = new_model_details
        
        print(f"Successfully switched to version: '{version_str}'")
        return {
            "message": f"Model '{MODEL_NAME}' successfully switched to version/stage: '{version_str}'.",
            "new_model_details": new_model_details
        }
    except (MlflowException, HTTPException) as e:
        # If MLFlow can't find the model/version, it raises an exception
        print(f"Error: Failed to load model version. {e}")
        # Re-raise if it's already an HTTPException, otherwise create one
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Failed to find or load model '{MODEL_NAME}' version/stage '{version_str}'. Details: {e}"
            )
    except Exception as e:
        print(f"An unexpected error occurred during model switching: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during model switching: {e}"
        )