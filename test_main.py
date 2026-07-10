import pytest
import pandas as pd  # <-- NEW IMPORT
from unittest.mock import MagicMock
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

# Import your FastAPI app
import main

@asynccontextmanager
async def dummy_lifespan(app):
    yield  

@pytest.fixture
def client():
    # 1. Fake Model
    fake_model = MagicMock()
    fake_model.predict.return_value = [0]                 
    fake_model.predict_proba.return_value = [[0.98, 0.02]] 

    # 2. Fake Store
    fake_store = MagicMock()
    
    mock_features = {f"V{i}": [0.0] for i in range(1, 29)}
    mock_features["Amount"] = [100.50]
    
    mock_response = MagicMock()
    
    # --- THE FIX IS HERE ---
    # We now explicitly tell the mock how to handle .to_df() calls!
    mock_response.to_df.return_value = pd.DataFrame(mock_features)
    mock_response.to_dict.return_value = mock_features
    
    fake_store.get_online_features.return_value = mock_response

    # 3. Inject Fakes
    main.ml_components["model"] = fake_model
    main.ml_components["store"] = fake_store
    
    main.app.router.lifespan_context = dummy_lifespan
    
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
    payload = {"transaction_id": 1001}
    response = client.post("/predict", json=payload)
    
    # PRO TIP: If a test fails with a 500 error again, uncomment the line 
    # below to see the EXACT error message FastAPI is throwing!
    # print("DEBUG ERROR:", response.json()) 
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["transaction_id"] == 1001
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert data["is_fraud"] == False
    assert data["fraud_probability"] == 0.02