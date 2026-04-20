import pandas as pd


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = [0, 12, 24, 48, 60, 100]
    labels = ["0-12", "12-24", "24-48", "48-60", "60+"]

    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    return df


def create_avg_monthly_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["avg_monthly_value"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    services = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["num_services"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

    return df


def create_protection_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_protection"] = (
        (df["OnlineSecurity"] == "Yes") |
        (df["DeviceProtection"] == "Yes") |
        (df["TechSupport"] == "Yes")
    ).astype(int)

    return df


def create_contract_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    risk_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0
    }

    df["contract_risk"] = df["Contract"].map(risk_map)

    return df


def create_payment_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df


def create_new_customer_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)

    return df


def create_high_value_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    threshold = df["MonthlyCharges"].median()

    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)

    return df


def create_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """
    df = create_tenure_groups(df)
    df = create_avg_monthly_value(df)
    df = create_service_count(df)
    df = create_protection_flag(df)
    df = create_contract_risk(df)
    df = create_payment_risk(df)
    df = create_new_customer_flag(df)
    df = create_high_value_flag(df)

    return df
