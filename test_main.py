from fastapi.testclient import TestClient # built on top of pytest and allows us to test our FastAPI endpoints without running the server. It simulates HTTP requests to our API.
from main import app

# Create a fake client to send requests to our API without starting the server
client = TestClient(app)

def test_predict_endpoint():
    # 1. Define dummy data
    payload = {
        "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0,
        "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0,
        "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0, "Amount": 100.50
    }
    
    # 2. Send a POST request to the /predict endpoint
    response = client.post("/predict", json=payload)
    
    # 3. Assert (verify) the results are what we expect
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "fraud_probability" in response.json()