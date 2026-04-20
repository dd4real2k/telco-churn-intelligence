import pandas as pd
from src.inference import assign_risk_tier, add_risk_tiers, prepare_single_customer_features


def test_assign_risk_tier():
    assert assign_risk_tier(0.80) == "High"
    assert assign_risk_tier(0.60) == "Medium"
    assert assign_risk_tier(0.20) == "Low"


def test_add_risk_tiers():
    df = pd.DataFrame({
        "churn_probability": [0.2, 0.6, 0.9]
    })
    scored = add_risk_tiers(df)
    assert scored["risk_tier"].tolist() == ["Low", "Medium", "High"]


def test_prepare_single_customer_features():
    df = pd.DataFrame([{
        "customerID": "CUST-1",
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 447.5
    }])

    featured = prepare_single_customer_features(df)
    assert "contract_risk" in featured.columns
    assert "num_services" in featured.columns
    assert "is_new_customer" in featured.columns
