import pandas as pd


def predict_churn_probability(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return churn probabilities and predicted classes.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    results = X.copy()
    results["churn_probability"] = probs
    results["predicted_churn"] = preds

    return results


def assign_risk_tier(probability: float) -> str:
    """
    Assign customer risk tier based on churn probability.
    """
    if probability >= 0.75:
        return "High"
    elif probability >= 0.50:
        return "Medium"
    else:
        return "Low"


def add_risk_tiers(df: pd.DataFrame, prob_col: str = "churn_probability") -> pd.DataFrame:
    """
    Add risk tier column to prediction dataframe.
    """
    df = df.copy()
    df["risk_tier"] = df[prob_col].apply(assign_risk_tier)
    return df

def prepare_scored_output(base_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach key business columns back to scored predictions.
    """
    output = scored_df.copy()
    for col in ["customerID", "Churn", "MonthlyCharges", "TotalCharges", "tenure", "Contract", "PaymentMethod"]:
        output[col] = base_df[col].values
    return output
