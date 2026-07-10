# serving/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI , HTTPException ,status
from pydantic import BaseModel
import pandas as pd
import mlflow
from feast import FeatureStore



# Global dictionary to hold our ML components
ml_components = {}

# Feast: knows how to fetch a transaction's features given its ID

# model = None  # filled at startup

@asynccontextmanager
async def lifespan(app: FastAPI):

    ml_components["store"] = FeatureStore(repo_path="./feature_repo/feature_repo")

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # global model
    ml_components["model"] = mlflow.sklearn.load_model(
        "models:/Logistic_Regression_Fraud_Registered/latest"
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness probe. If the server can respond to this, it's alive.
    """
    return {"status": "healthy"}

@app.get("/ready", tags=["System"])
def readiness_check():
    """
    Readiness probe. Checks if MLflow and Feast are fully loaded into memory.
    """
    # Check if the keys exist and are not None
    store_loaded = ml_components.get("store") is not None
    model_loaded = ml_components.get("model") is not None

    if store_loaded and model_loaded:
        return {"status": "ready"}
    else:
        # Return a 503 Service Unavailable if they aren't loaded yet
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning components are still initializing."
        )
    
# The caller now sends ONLY an id — Feast supplies the 29 features
class Transaction(BaseModel):
    transaction_id: int

@app.post("/predict")
def predict_fraud(request: Transaction):
    try:
        feast_request_features = [
            "transaction_features:V1",
            "transaction_features:V2",
            "transaction_features:V3",
            "transaction_features:V4",
            "transaction_features:V5",
            "transaction_features:V6",
            "transaction_features:V7",
            "transaction_features:V8",
            "transaction_features:V9",
            "transaction_features:V10",
            "transaction_features:V11",
            "transaction_features:V12",
            "transaction_features:V13",
            "transaction_features:V14",
            "transaction_features:V15",
            "transaction_features:V16",
            "transaction_features:V17",
            "transaction_features:V18",
            "transaction_features:V19",
            "transaction_features:V20",
            "transaction_features:V21",
            "transaction_features:V22",
            "transaction_features:V23",
            "transaction_features:V24",
            "transaction_features:V25",
            "transaction_features:V26",
            "transaction_features:V27",
            "transaction_features:V28",
            "transaction_features:Amount"
        ]
        
        model_expected_columns = [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]
        
        # 1. Look up this transaction's features in Feast's online store
        feature_vector = ml_components["store"].get_online_features(
            features=feast_request_features,
            entity_rows=[{"transaction_id": request.transaction_id}]
        ).to_df()
        
        # Convert to DataFrame
        df_features = pd.DataFrame(feature_vector)

        # 4. ENFORCE COLUMN ORDER using the stripped column names
        df_features = df_features[model_expected_columns]

        # 3. Predict + probability
        prediction = ml_components["model"].predict(df_features)
        probability = ml_components["model"].predict_proba(df_features)[0][1]

        return {
            "transaction_id": request.transaction_id,
            "is_fraud": bool(prediction[0]),
            "fraud_probability": round(float(probability), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))