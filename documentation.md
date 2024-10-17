## Iris Flower Classification API - Readme

**Welcome to the Iris Flower Classification API!**

This API allows you to predict the species of an iris flower based on its sepal and petal measurements. It's built using FastAPI and leverages a pre-trained Support Vector Classifier (SVC) model.

### Features

* **Predict Iris Species:**  Classify an iris flower into one of three species: setosa, versicolor, or virginica. ([/predict](/predict))
* **Batch Prediction:**  Classify multiple iris flowers in a single request. ([/predict_batch](/predict_batch))
* **Random Prediction:** Get a random iris flower prediction along with its features. ([/predict_random](/predict_random))
* **Health Check:** Verify the API is running smoothly. ([/health](/health))
* **Model Information:** Learn more about the underlying model used for prediction. ([/model_info](/model_info))
* **Workload Simulation:** Simulate workload for testing purposes. ([/simulate_workload](/simulate_workload))

### Installation

1. **Prerequisites:** Ensure you have Python 3.6 or later installed on your system. You can check by running `python --version` in your terminal.
2. **Install dependencies:** Clone or download this repository and run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Usage

**1. Start the API server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

This will start the API server on port 8000 by default. You can access the API endpoints using tools like Postman, curl, or directly through your browser.

**2. Prediction endpoints:**

- **Predict single iris:**

```
POST /predict
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "prediction": "setosa",
  "code_signature": "..."
}
```

- **Batch prediction:**

```
POST /predict_batch
Content-Type: application/json

{
  "features": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 4.9,
      "sepal_width": 3.0,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  ]
}
```

**Response:**

```json
{
  "predictions": ["setosa", "setosa"],
  "code_signature": "..."
}
```

- **Random prediction:**

```
GET /predict_random
```

**Response:**

```json
{
  "random_features": {
    "sepal_length": 5.32,
    "sepal_width": 3.18,
    "petal_length": 1.54,
    "petal_width": 0.42
  },
  "prediction": "versicolor",
  "code_signature": "..."
}
```

**3. Other endpoints:**

* **Health Check:** Access `/health` for a simple health check response.
* **Model Information:** Access `/model_info` to learn about the model used for prediction.
* **Workload Simulation:** Use `/simulate_workload?seconds={duration}` to simulate workload for a specified duration (in seconds). This is helpful for testing API performance under load.

### Contributing

We welcome contributions to this project! Feel free to fork the repository, make changes, and submit pull requests.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
