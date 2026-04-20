import pandas as pd
from src.features import create_churn_features


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


def prepare_single_customer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering for incoming raw customer data.
    """
    df = input_df.copy()
    df = create_churn_features(df)
    return df


def score_single_customer(model, input_df: pd.DataFrame) -> dict:
    """
    Score a single customer record and return prediction output.
    """
    featured_df = prepare_single_customer_features(input_df)

    X = featured_df.copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    probability = float(probs[0])
    prediction = int(preds[0])
    risk_tier = assign_risk_tier(probability)

    return {
        "churn_probability": round(probability, 4),
        "predicted_churn": prediction,
        "predicted_label": "Yes" if prediction == 1 else "No",
        "risk_tier": risk_tier
    }
