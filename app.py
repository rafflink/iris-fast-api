from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import numpy as np
import joblib
import uvicorn
import logging
import random
import time
import hashlib
from fastapi.middleware.cors import CORSMiddleware

# Load the model
try:
    model = joblib.load('svc_model.pkl')
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    raise RuntimeError("Model loading failed. Ensure 'svc_model.pkl' is present.")

# Create FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="API for classifying iris flowers based on sepal and petal measurements.",
    version="1.0.0",
    contact={
        "name": "Samurai",
    }
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unique identifier for the code version
CODE_SIGNATURE = hashlib.md5("IrisAPI_v1.0.0_Hackathon".encode()).hexdigest()

# Define the request schema for single prediction
class IrisFeatures(BaseModel):
    """
    Data model for iris flower features used in the prediction.

    Attributes:
        sepal_length (float): Sepal length in cm (0 < sepal_length <= 8.0)
        sepal_width (float): Sepal width in cm (0 < sepal_width <= 4.5)
        petal_length (float): Petal length in cm (0 < petal_length <= 7.0)
        petal_width (float): Petal width in cm (0 < petal_width <= 2.5)
    """

    sepal_length: float = Field(..., gt=0, le=8.0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=4.5, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=7.0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=2.5, description="Petal width in cm")

    @validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    def validate_positive_values(cls, value):
        """
        Validator to ensure that all feature values are greater than zero.

        Args:
            value (float): The input feature value.

        Returns:
            float: The validated feature value.
        """
        if value <= 0:
            raise ValueError("Measurements must be greater than zero.")
        return value

# Define the request schema for batch prediction
class BatchIrisFeatures(BaseModel):
    """
    Data model for a batch of iris flower feature sets used in batch predictions.

    Attributes:
        features (List[IrisFeatures]): A list of iris feature sets.
    """
    features: List[IrisFeatures]

# Define a mapping for model predictions
prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Predict endpoint for a single flower
@app.post("/predict", summary="Predict Iris Species", tags=["Prediction"])
async def predict(features: IrisFeatures):
    """
    Predict the species of an iris flower based on its sepal and petal measurements.

    Args:
        features (IrisFeatures): The sepal and petal measurements of the flower.

    Returns:
        dict: The predicted species and code signature.
    """
    try:
        # Log incoming request
        logging.info(f"Received prediction request: {features.dict()}")

        # Prepare data for prediction
        input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])

        # Perform prediction
        prediction = model.predict(input_data)
        return {
            "prediction": prediction_map[int(prediction[0])],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Batch prediction endpoint
@app.post("/predict_batch", summary="Batch Predict Iris Species", tags=["Prediction"])
async def predict_batch(batch_features: BatchIrisFeatures):
    """
    Predict the species of multiple iris flowers in a batch.

    Args:
        batch_features (BatchIrisFeatures): A list of feature sets for multiple flowers.

    Returns:
        dict: A list of predicted species and the code signature.
    """
    try:
        # Log incoming batch request
        logging.info(f"Received batch prediction request with {len(batch_features.features)} features sets")

        # Prepare data for batch prediction
        input_data = np.array([
            [features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]
            for features in batch_features.features
        ])

        # Perform batch prediction
        predictions = model.predict(input_data)
        return {
            "predictions": [prediction_map[int(pred)] for pred in predictions],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

# Predict endpoint with randomly generated features
@app.get("/predict_random", summary="Get Random Iris Prediction", tags=["Prediction"])
async def predict_random():
    """
    Generate random iris flower features and predict its species.

    Returns:
        dict: The randomly generated features, predicted species, and code signature.
    """
    try:
        # Generate random features
        random_features = IrisFeatures(
            sepal_length=random.uniform(4.0, 8.0),
            sepal_width=random.uniform(2.0, 4.5),
            petal_length=random.uniform(1.0, 7.0),
            petal_width=random.uniform(0.1, 2.5)
        )

        # Log the random features generated
        logging.info(f"Generated random features: {random_features}")

        # Prepare data for prediction
        input_data = np.array([[random_features.sepal_length, random_features.sepal_width, random_features.petal_length, random_features.petal_width]])

        # Perform prediction
        prediction = model.predict(input_data)
        return {
            "random_features": random_features.dict(),
            "prediction": prediction_map[int(prediction[0])],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Random prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Random prediction failed: {e}")

# Health check endpoint
@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check():
    """
    Perform a health check of the API.

    Returns:
        dict: API health status and code signature.
    """
    return {
        "status": "Healthy",
        "code_signature": CODE_SIGNATURE
    }

# Endpoint to retrieve model information
@app.get("/model_info", summary="Get Model Information", tags=["Info"])
async def model_info():
    """
    Retrieve information about the model used for iris classification.

    Returns:
        dict: Model details including its type, version, and description.
    """
    return {
        "model": "Support Vector Classifier (SVC)",
        "version": "1.0",
        "description": "Model trained on the Iris dataset for species classification.",
        "code_signature": CODE_SIGNATURE
    }

# Endpoint to simulate workload for testing latency
@app.get("/simulate_workload", summary="Simulate Workload", tags=["Testing"])
async def simulate_workload(seconds: Optional[int] = 1):
    """
    Simulate server workload for a specified duration to test latency.

    Args:
        seconds (Optional[int]): Number of seconds to simulate the workload.

    Returns:
        dict: Success message and code signature after workload simulation.
    """
    try:
        # Log workload simulation request
        logging.info(f"Simulating workload for {seconds} seconds")

        # Simulate workload by sleeping
        time.sleep(seconds)
        return {
            "message": f"Successfully simulated workload for {seconds} seconds",
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logging.error(f"Workload simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Workload simulation failed: {e}")

