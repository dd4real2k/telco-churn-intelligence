import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score

from src.config import FEATURED_DATA_PATH, RANDOM_STATE


def load_feature_data(path=FEATURED_DATA_PATH):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop(columns=["Churn", "ChurnFlag", "customerID"])
    y = df["ChurnFlag"]

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    return preprocessor


def build_models():
    models = {
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    preprocessor = build_preprocessor(X_train)
    models = build_models()

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            "model": pipeline,
            "auc": auc
        }

        print(f"{name} AUC: {auc:.4f}")

    return results

def select_best_model(results):
    best_model_name = max(results, key=lambda x: results[x]["auc"])
    best_model = results[best_model_name]["model"]
    best_auc = results[best_model_name]["auc"]

    print(f"Best model: {best_model_name} | AUC: {best_auc:.4f}")
    return best_model
