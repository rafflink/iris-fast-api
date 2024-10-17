## Iris Flower Classification API - Readme

**Welcome to the Iris Flower Classification API!**

This API predicts the species of an iris flower based on its sepal and petal measurements. It's built using FastAPI and powered by a pre-trained Support Vector Classifier (SVC) model.

### Features

- **Predict Iris Species:** Classifies an iris flower into one of three species: *Setosa*, *Versicolor*, or *Virginica*. ([/predict](/predict))
- **Batch Prediction:** Classify multiple iris flowers in a single request. ([/predict_batch](/predict_batch))
- **Random Prediction:** Provides a random iris flower's measurements and predicts its species. ([/predict_random](/predict_random))
- **Health Check:** Confirms that the API is running as expected. ([/health](/health))
- **Model Information:** Retrieves details about the model used for predictions. ([/model_info](/model_info))
- **Workload Simulation:** Simulates load for performance testing. ([/simulate_workload](/simulate_workload))

### Usage

#### 1. Start the API Server

To start the API server on port 8000, run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
You can interact with the API via tools like Postman, `curl`, or a browser.

#### 2. API Endpoints

- **Single Flower Prediction:**
   Submit a flower's sepal and petal measurements to predict its species.
   ```http
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

- **Batch Prediction:**
   Predict species for multiple iris flowers in one request.
   ```http
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

- **Random Prediction:**
   Get random sepal and petal measurements and the corresponding species prediction.
   ```http
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

#### 3. Other Endpoints

- **Health Check:**  
   Access `/health` to verify if the API is operational.

- **Model Information:**  
   View details about the prediction model via `/model_info`.

- **Workload Simulation:**  
   Simulate workload with `/simulate_workload?seconds={duration}` to test performance.

### Contributing

Contributions are welcome! Feel free to fork the repository, submit your changes, and create pull requests.

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
