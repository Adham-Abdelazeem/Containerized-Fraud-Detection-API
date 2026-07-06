from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
from feast import FeatureStore


store = FeatureStore(repo_path="../feature_repo")

# Keep the model container global but uninitialized
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set this to wherever your MLflow registry is hosted
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # Example: Local MLflow server
    # OR mlflow.set_tracking_uri("sqlite:///mlflow.db") if using a local sqlite backend

    global model
    # The model is loaded ONLY when the server starts up, not when imported
    model = mlflow.sklearn.load_model("models:/Logistic_Regression_Fraud_Registered/latest")
    yield
    # Clean up actions (if any) go here when the server shuts down

app = FastAPI(lifespan=lifespan)

class Transaction(BaseModel):
    transaction_id: int

# 4. Create the POST endpoint
@app.post("/predict") # this is the endpoint - a decorator
def predict_fraud(request: Transaction): # variable name : Type Hint
    # Fetch features from Feast using the provided transaction_id
    feature_vector = store.get_online_features(
        feature_service=store.get_feature_service("fraud_model_v1"),
        entity_rows=[{"transaction_id": request.transaction_id}],
    ).to_df()

    # Drop metadata columns (entity keys) before feeding to the Scikit-learn model
    data_df = feature_vector.drop(columns=["transaction_id"])
    
    # Make the prediction (returns an array like [0] or [1])
    prediction = model.predict(data_df)
    
    # Get the probability of it being fraud (class 1)
    probability = model.predict_proba(data_df)[0][1]
    
    # Return a JSON response
    return {
        "transaction_id": request.transaction_id,
        "is_fraud": bool(prediction[0]),
        "fraud_probability": round(float(probability), 4)
    }