from pathlib import Path
import sys

# Ensure project root is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import MODELS_DIR
from src.inference import score_single_customer

app = FastAPI(
    title="Telco Churn Intelligence API",
    description="API for predicting customer churn risk",
    version="1.0.0"
)

# Load trained model once on startup
model = joblib.load(MODELS_DIR / "churn_model.joblib")


class CustomerInput(BaseModel):
    customerID: str = "CUST-0001"
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {
        "message": "Welcome to the Telco Churn Intelligence API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    input_df = pd.DataFrame([customer.dict()])
    result = score_single_customer(model, input_df)

    return {
        "customerID": customer.customerID,
        **result
    }
