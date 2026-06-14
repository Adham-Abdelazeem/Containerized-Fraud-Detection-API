# Containerized Fraud Detection API

[![CI Pipeline](https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API/actions/workflows/ci.yml/badge.svg)](https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## Project Overview
This project is an end-to-end Machine Learning API designed to predict fraudulent credit card transactions in real-time.

Instead of just training a model in a Jupyter Notebook, this project demonstrates a complete **MLOps pipeline**. It takes a trained Scikit-Learn model, wraps it in a high-performance FastAPI server, containerizes the application using Docker for reproducible deployments, and utilizes GitHub Actions for Continuous Integration (CI).

## Tech Stack
* **Machine Learning:** Scikit-learn, Pandas, Joblib
* **Experiment Tracking:** MLflow (parameter logging, metric tracking, model registry)
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Containerization:** Docker
* **CI/CD & Testing:** GitHub Actions, Pytest

## Architecture & Features

1. **Model Training with MLflow:** A Logistic Regression model trained on a highly imbalanced, real-world credit card fraud dataset. Training is wrapped in an MLflow run that logs hyperparameters (`max_iter`, `class_weight`), evaluation metrics (accuracy, precision, sensitivity/recall, F1), the model artifact (registered as `Logistic_Regression_Fraud_Registered`), and the inferred input/output signature — all visible in the MLflow UI at `http://localhost:5000`.
2. **REST API:** A FastAPI endpoint (`/predict`) that accepts transaction features via JSON, validates the payload using Pydantic, and returns a fraud probability score.
3. **Dockerized Environment:** The entire application is packaged into a lightweight Docker image, ensuring it runs identically across local machines and cloud servers without dependency conflicts.
4. **Automated CI Pipeline:** Every push to the `main` branch triggers a GitHub Actions workflow that automatically runs unit tests and verifies the Docker build process.

---

## How to Run Locally

### Option 1: Using Docker (Recommended)
You do not need Python installed on your machine to run this, only Docker.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API.git
   cd Containerized-Fraud-Detection-API
   ```
2. **Build the Docker image:**
   ```bash
   docker build -t fraud-api .
   ```
3. **Run the container:**
   ```bash
   docker run -p 8000:8000 fraud-api
   ```
4. **Access the API:** Open your browser and navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI.

### Option 2: Using Python Virtual Environment
1. Clone the repo and navigate to the directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   fastapi dev main.py
   ```

---

## Example API Usage

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
  "V1": -1.3, "V2": 0.2, "V3": 1.5, "V4": 0.4, "V5": -0.5,
  "V6": 0.1, "V7": 0.2, "V8": 0.1, "V9": 0.8, "V10": -0.2,
  "V11": 0.0, "V12": 0.5, "V13": -0.1, "V14": 0.3, "V15": 1.2,
  "V16": 0.5, "V17": -0.4, "V18": 0.1, "V19": 0.2, "V20": 0.1,
  "V21": -0.1, "V22": 0.2, "V23": -0.3, "V24": 0.4, "V25": 0.1,
  "V26": 0.2, "V27": -0.1, "V28": 0.0,
  "Amount": 150.00
}'
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0214
}
```

## Future Improvements
* Deploy the Docker container to a cloud provider (e.g., AWS AppRunner, Google Cloud Run, or Render).
* Implement more advanced ML algorithms (e.g., XGBoost) and hyperparameter tuning to improve recall on fraudulent cases.
* Add API key authentication for secure endpoint access.
