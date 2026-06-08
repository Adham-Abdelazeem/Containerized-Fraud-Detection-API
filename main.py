from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Fraud Detection API", description="API to predict fraudulent credit card transactions")

# 2. Load the trained model into memory
# We do this outside the endpoint so it only loads ONCE when the server starts, not every time a request comes in.
model = joblib.load('model/fraud_model.joblib')

# 3. Define the Pydantic data model (The Security Guard of our API)
# This strictly defines what the incoming JSON must look like. 
# Our training data had features V1 through V28, plus 'Amount'.
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# 4. Create the POST endpoint
@app.post("/predict") # this is the endpoint - a decorator
def predict_fraud(transaction: Transaction): # variable name : Type Hint
    # Convert the validated Pydantic object into a Python dictionary
    data_dict = transaction.model_dump()
    
    # Convert the dictionary into a pandas DataFrame (which Scikit-learn expects)
    # We wrap data_dict in a list so it becomes a single row in the DataFrame
    data_df = pd.DataFrame([data_dict])
    
    # Make the prediction (returns an array like [0] or [1])
    prediction = model.predict(data_df)
    
    # Get the probability of it being fraud (class 1)
    probability = model.predict_proba(data_df)[0][1]
    
    # Return a JSON response
    return {
        "is_fraud": bool(prediction[0]),
        "fraud_probability": round(float(probability), 4)
    }

#I have:    a clean, validated Transaction object
#I need:    a prediction from my sklearn model
#sklearn expects:  a pandas DataFrame
#I have:           a Pydantic object
#So I need to:     Pydantic object → dict → DataFrame → model → result

# this decoder is like 
# def predict_fraud(transaction):
#   #code here
# Python passes the function directly into the decorator tool
# predict_fraud = app.post("/predict")(predict_fraud)