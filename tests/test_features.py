import pandas as pd
from src.features import (
    create_tenure_groups,
    create_service_count,
    create_protection_flag,
    create_contract_risk,
    create_payment_risk,
    create_churn_features
)


def sample_df():
    return pd.DataFrame({
        "customerID": ["C1", "C2"],
        "tenure": [5, 30],
        "MonthlyCharges": [80.0, 60.0],
        "TotalCharges": [400.0, 1800.0],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"]
    })


def test_create_tenure_groups():
    df = create_tenure_groups(sample_df())
    assert "tenure_group" in df.columns


def test_create_service_count():
    df = create_service_count(sample_df())
    assert "num_services" in df.columns
    assert df["num_services"].iloc[0] >= 0


def test_create_protection_flag():
    df = create_protection_flag(sample_df())
    assert "has_protection" in df.columns
    assert set(df["has_protection"].unique()).issubset({0, 1})


def test_create_contract_risk():
    df = create_contract_risk(sample_df())
    assert df["contract_risk"].tolist() == [2, 0]


def test_create_payment_risk():
    df = create_payment_risk(sample_df())
    assert df["is_electronic_check"].tolist() == [1, 0]


def test_create_churn_features_adds_expected_columns():
    df = create_churn_features(sample_df())
    expected_cols = {
        "tenure_group",
        "avg_monthly_value",
        "num_services",
        "has_protection",
        "contract_risk",
        "is_electronic_check",
        "is_new_customer",
        "is_high_value"
    }
    assert expected_cols.issubset(df.columns)
