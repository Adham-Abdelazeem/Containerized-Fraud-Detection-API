# Containerized Fraud Detection API

[![CI Pipeline](https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API/actions/workflows/ci.yml/badge.svg)](https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)
![Feast](https://img.shields.io/badge/Feast-Feature_Store-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Hardened-blue.svg)

## Project Overview
This project is an end-to-end Machine Learning API designed to predict fraudulent credit card transactions in real-time.

Instead of just training a model in a Jupyter Notebook, this project demonstrates a complete, production-grade **MLOps pipeline**. It integrates a **Feature Store** (Feast) for data management, an **Experiment Tracker and Model Registry** (MLflow), and a high-performance **FastAPI** serving layer, all wrapped in a security-hardened Docker container.

## Tech Stack
* **Machine Learning:** Scikit-learn, Pandas, Joblib
* **Feature Store:** Feast (Historical offline training & SQLite online serving)
* **Experiment Tracking & Registry:** MLflow (Parameter logging, metric tracking, model registry)
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Containerization:** Docker (Multi-stage builds, non-root user execution)
* **CI/CD & Testing:** GitHub Actions, Pytest

## Architecture & Features

1. **Feature Store Integration (Feast):** Eliminates training-serving skew. The model trains on historical offline Parquet data, while the FastAPI server fetches ultra-low-latency features from a materialized SQLite online store using just a `transaction_id`.
2. **Model Training & Registry (MLflow):** A Logistic Regression model trained on highly imbalanced credit card fraud data. Training runs log hyperparameters, evaluation metrics (accuracy, precision, recall, F1), and register the final model artifact to be automatically pulled by the serving layer.
3. **Production-Ready REST API:** A FastAPI endpoint (`/predict`) that dynamically fetches features and returns a fraud probability score. It includes `/health` (liveness) and `/ready` (readiness) probes to ensure safe deployment in container orchestration systems (like Kubernetes or ECS).
4. **Hardened Docker Environment:** The application is packaged using security best practices, utilizing a multi-stage build to reduce image size and running as a restricted, non-root user (`appuser`) to minimize attack surfaces.

---

## The Full MLOps Lifecycle (How to Run Locally)

Because this is a full MLOps pipeline, you must initialize the Feature Store and train the model before serving predictions. 

### 1. Setup Environment
Clone the repository and install dependencies:
` ` `bash
git clone [https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API.git](https://github.com/Adham-Abdelazeem/Containerized-Fraud-Detection-API.git)
cd Containerized-Fraud-Detection-API
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
` ` `

### 2. Initialize the Feature Store (Feast)
Create the physical SQLite database and materialize the offline Parquet data into the online store for fast retrieval:
` ` `bash
# 1. Create the infrastructure (online_store.db)
feast -c ./feature_repo/feature_repo apply

# 2. Push historical data to the online store
python force_materialize.py
` ` `

### 3. Train & Register the Model (MLflow)
Run the training script. This will pull historical features from Feast, train the Logistic Regression model, and register it in local MLflow:
` ` `bash
python train.py
` ` `

### 4. Start the API Server
#### Option A: Using FastAPI directly
` ` `bash
uvicorn serving.main:app --host 0.0.0.0 --port 8000
` ` `
#### Option B: Using the Hardened Docker Container
` ` `bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
` ` `

---

## Example API Usage

Because Feast handles feature retrieval dynamically, the client only needs to pass a `transaction_id`!

**1. Check Server Readiness:**
` ` `bash
curl [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready)
` ` `
*(Returns `{"status": "ready"}` when MLflow and Feast are fully loaded into memory).*

**2. Make a Prediction:**
` ` `bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 1001}'
` ` `

**Response:**
` ` `json
{
  "transaction_id": 1001,
  "is_fraud": false,
  "fraud_probability": 0.0214
}
` ` `
