"""
Serve fraud detection model from MLflow Model Registry.

This version loads the @champion model from MLflow, which means:
- Always serves the latest @champion model
- Can roll back by changing the @champion alias
- No manual file copying needed
"""
import pickle
import os
from fastapi import FastAPI, HTTPException
from src.model_loader import load_model
from src.models import PredictionResponse, Transaction, ValidationErrorResponse
from src.data_validation import validate_transaction

model, encoder = load_model()

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Encoder loaded successfully!")

app = FastAPI(
    title="Fraud Detection API (MLflow)",
    description="""
    Fraud detection API  with input validation that loads models from MLflow Model Registry.
    
    All inputs are validated before prediction:
    - amount: Must be positive and below $50,000
    - hour: Must be 0-23
    - day_of_week: Must be 0-6
    - merchant_category: Must be one of: grocery, restaurant, retail, online, travel
    
    Invalid inputs return HTTP 400 with detailed error messages.
    
    This version always serves the model with the @champion alias.
    To update the model:
    1. Train a new model with train_mlflow.py
    2. Compare metrics in MLflow UI
    3. Promote the best model to Production
    4. Restart this API
    
    To roll back: Move the @champion alias to a previous version in MLflow UI.
    """,
    version="4.0.0"
)

@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ValidationErrorResponse}})
def predict(tx: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Input is validated before prediction. Invalid inputs return HTTP 400.
    """
    data = tx.dict()
    
    # VALIDATE INPUT BEFORE MAKING PREDICTION
    validation = validate_transaction(data)
    
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Validation failed",
                "errors": validation["errors"],
                "input": data
            }
        )
    
    # Input is valid - make prediction
    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        validation_passed=True
    )

@app.get("/health")
def health():
    return {"status": "healthy", "validation": "enabled"}

@app.get("/model-info")
def model_info():
    """Get information about the currently loaded model."""
    return {
        "registry": "MLflow",
        "model_name": "fraud-detection-model",
        "alias": "champion",
        "tracking_uri": "http://localhost:5000"
    }