from pydantic import BaseModel, Field

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", example=150.00)
    hour: int = Field(..., description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", example=3)
    merchant_category: str = Field(..., description="Merchant type", example="online")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    validation_passed: bool = True

class ValidationErrorResponse(BaseModel):
    detail: dict