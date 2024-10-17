# Instructions for Setting Up and Running the Iris Flower Classification API

### 1. Setting Up the Environment

- **Python Installation:** Ensure that Python 3.6 or later is installed on your system. Verify this by running:
   ```bash
   python --version
   ```
  
- **Virtual Environment (Recommended):**  
   Create a virtual environment to manage project dependencies. You can use `venv` or `conda`. To create and activate a virtual environment with `venv`, follow these steps:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   .\env\Scripts\activate   # For Windows
   ```

### 2. Creating the Project Structure

- **Create a Project Directory:**
   ```bash
   mkdir iris_classification_api
   cd iris_classification_api
   ```

- **Main Python File:**
   Inside the project directory, create a Python file named `main.py` that will contain the API code.

### 3. Installing Dependencies

- **Create `requirements.txt`:**  
  Inside your project folder, create a file named `requirements.txt` and add the following dependencies:
   ```
   fastapi
   pydantic
   numpy
   joblib
   uvicorn
   logging
   random
   time
   hashlib
   ```

- **Install Dependencies:**  
  Run the following command to install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Writing the Code

- **Open `main.py`:**  
  Use your preferred code editor or IDE to open the `main.py` file.

### 5. Importing Libraries

- At the top of `main.py`, import the required libraries. The libraries serve specific purposes such as managing FastAPI, loading models, and generating predictions.

### 6. Loading the Model

- **Model Loading:**  
  Use the `joblib.load()` function to load your pre-trained Support Vector Classifier (SVC) model from a file (e.g., `svc_model.pkl`). Ensure that the model file is present in your project directory.

  Example:
  ```python
  from joblib import load

  model = load("svc_model.pkl")
  ```

- **Error Handling:**  
  Use a `try-except` block to handle errors in case the model file cannot be loaded.

### 7. Creating the FastAPI App

- **Initialize FastAPI:**  
  Create an instance of the FastAPI application and add any metadata like the title, description, and contact info for better documentation:
   ```python
   from fastapi import FastAPI

   app = FastAPI(
       title="Iris Flower Classification API",
       description="API for predicting iris flower species using a pre-trained SVC model.",
       version="1.0.0"
   )
   ```

### 8. Defining Data Models (Optional)

- **Pydantic Models:**  
  Use Pydantic models to define the data structure for incoming requests and outgoing responses, ensuring data validation and clear API documentation.

  Example:
  ```python
  from pydantic import BaseModel

  class IrisFeatures(BaseModel):
      sepal_length: float
      sepal_width: float
      petal_length: float
      petal_width: float
  ```

### 9. Unique Code Signature (Optional)

- **Code Versioning:**  
  Create a unique identifier for the code using `hashlib.md5()`, which will help you track code changes and versions:
  ```python
  import hashlib

  CODE_SIGNATURE = hashlib.md5(b"your-unique-code").hexdigest()
  ```

### 10. API Endpoints

- **Define Endpoints:**  
  Use decorators like `@app.post` and `@app.get` to define API endpoints. Here are a few examples:

  - **Predict a Single Iris Flower:**
    ```python
    @app.post("/predict")
    async def predict_iris(features: IrisFeatures):
        # Prediction logic
    ```

  - **Batch Prediction:**
    ```python
    @app.post("/predict_batch")
    async def predict_batch(batch_features: List[IrisFeatures]):
        # Batch prediction logic
    ```

  - **Health Check:**
    ```python
    @app.get("/health")
    async def health_check():
        return {"status": "Healthy"}
    ```

### 11. Running the API

- **Uvicorn Server:**  
  At the bottom of your code, include a block to run the FastAPI app using Uvicorn:
  ```python
  if __name__ == "__main__":
      import uvicorn
      uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
  ```

- **Start the Server:**  
  Run the following command to start the API server:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

### Additional Notes

- **Customizing:**  
  Adjust the name of your model file (`svc_model.pkl`) if it differs, and modify API endpoints or logic as needed.
  
- **Logging:**  
  Use Python’s `logging` module for debugging or recording important events during the API's operation.
