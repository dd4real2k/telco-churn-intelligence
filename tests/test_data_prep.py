import pandas as pd
from src.data_prep import clean_total_charges, clean_churn_target, clean_telco_data


def test_clean_total_charges_converts_blank_to_numeric():
    df = pd.DataFrame({
        "TotalCharges": ["100.5", " ", "250.0"]
    })

    cleaned = clean_total_charges(df)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert pd.isna(cleaned.loc[1, "TotalCharges"])


def test_clean_churn_target_creates_binary_flag():
    df = pd.DataFrame({
        "Churn": ["Yes", "No", "Yes"]
    })

    cleaned = clean_churn_target(df)

    assert "ChurnFlag" in cleaned.columns
    assert cleaned["ChurnFlag"].tolist() == [1, 0, 1]


def test_clean_telco_data_keeps_expected_columns():
    df = pd.DataFrame({
        "customerID": ["A1", "A2"],
        "SeniorCitizen": [0, 1],
        "tenure": [0, 10],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [" ", "700.0"],
        "Churn": ["No", "Yes"]
    })

    cleaned = clean_telco_data(df)

    assert "ChurnFlag" in cleaned.columns
    assert "TotalCharges" in cleaned.columns
    assert cleaned.loc[0, "TotalCharges"] == 0
