import pytest
from unittest.mock import MagicMock
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

# Import your FastAPI app
# (If main.py is inside a 'serving' folder, use: import serving.main as main)
import main

# 1. Create a completely empty lifespan context manager for tests
@asynccontextmanager
async def dummy_lifespan(app):
    yield  # Skips execution of the real MLflow and Feast loading entirely

@pytest.fixture
def client():
    # 2. Create a mock MLflow model
    # It must mock BOTH predict() and predict_proba() because your API uses both
    fake_model = MagicMock()
    fake_model.predict.return_value = [0]                 # Predicts: Not Fraud
    fake_model.predict_proba.return_value = [[0.98, 0.02]] # 2% chance of fraud

    # 3. Create a mock Feast Feature Store
    fake_store = MagicMock()
    
    # Create a dummy dictionary representing the 29 features Feast normally returns
    mock_features = {f"V{i}": [0.0] for i in range(1, 29)}
    mock_features["Amount"] = [100.50]
    
    # Chain the mocks so get_online_features().to_dict() returns our dummy data
    mock_response = MagicMock()
    mock_response.to_dict.return_value = mock_features
    fake_store.get_online_features.return_value = mock_response

    # 4. Inject both fakes into main.py's global dictionary
    main.ml_components["model"] = fake_model
    main.ml_components["store"] = fake_store
    
    # 5. Hijack FastAPI's internal router to use our dummy lifespan instead
    main.app.router.lifespan_context = dummy_lifespan
    
    # 6. Spin up the client safely without hitting any external databases
    with TestClient(main.app) as ac:
        yield ac

# ==========================================
# TESTS
# ==========================================

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ready_check(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_predict_endpoint(client):
    # 1. Define dummy data (Now it ONLY needs the transaction_id!)
    payload = {
        "transaction_id": 1001
    }
    
    # 2. Send a POST request to the /predict endpoint
    response = client.post("/predict", json=payload)
    
    # 3. Assert (verify) the results are what we expect
    assert response.status_code == 200
    data = response.json()
    
    assert data["transaction_id"] == 1001
    assert "is_fraud" in data
    assert "fraud_probability" in data
    
    # Verify the specific fake data we injected was used
    assert data["is_fraud"] == False
    assert data["fraud_probability"] == 0.02