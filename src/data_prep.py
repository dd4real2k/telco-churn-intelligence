import pandas as pd
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw telco churn dataset.
    """
    return pd.read_csv(path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for consistency.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalCharges column by converting blanks to NaN,
    then cast to numeric.
    """
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SeniorCitizen to categorical labels for readability.
    """
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def clean_churn_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Churn to binary target.
    """
    df = df.copy()
    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def handle_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing TotalCharges values.
    In this dataset, missing values usually occur for customers with tenure = 0.
    We will fill them with 0 for consistency.
    """
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    df = df.copy()
    df = df.drop_duplicates()
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the telco churn dataset.
    """
    df = standardize_column_names(df)
    df = clean_total_charges(df)
    df = clean_senior_citizen(df)
    df = clean_churn_target(df)
    df = handle_missing_total_charges(df)
    df = remove_duplicates(df)
    return df


def save_clean_data(df: pd.DataFrame, path=CLEANED_DATA_PATH) -> None:
    """
    Save cleaned dataframe to CSV.
    """
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_telco_data(df_raw)
    save_clean_data(df_clean)
    print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")
    print(f"Final shape: {df_clean.shape}")
