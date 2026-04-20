import pandas as pd
import matplotlib.pyplot as plt


def get_logistic_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from a trained logistic regression pipeline.
    Returns a dataframe of features and coefficients.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

    return importance_df


def plot_top_coefficients(importance_df: pd.DataFrame, top_n: int = 15, save_path=None):
    """
    Plot top positive and negative logistic regression coefficients.
    """
    top_positive = importance_df.sort_values("coefficient", ascending=False).head(top_n)
    top_negative = importance_df.sort_values("coefficient", ascending=True).head(top_n)

    plot_df = pd.concat([top_negative, top_positive])

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.title("Top Positive and Negative Churn Drivers")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
