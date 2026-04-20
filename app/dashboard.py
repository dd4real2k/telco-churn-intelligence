from pathlib import Path
import sys

# Ensure project root is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURED_DATA_PATH, MODELS_DIR
from src.inference import score_single_customer


st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📉",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(FEATURED_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "churn_model.joblib")


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main():
    st.title("📉 Telco Churn Intelligence Dashboard")
    st.markdown(
        """
        An end-to-end churn analytics dashboard for identifying at-risk customers,
        estimating revenue exposure, and supporting retention decisions.
        """
    )

    df = load_data()
    model = load_model()

    tab1, tab2, tab3 = st.tabs([
        "Executive Overview",
        "Churn Predictor",
        "High-Risk Customers"
    ])

    # ---------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # ---------------------------
    with tab1:
        st.subheader("Executive Overview")

        total_customers = len(df)
        churn_rate = df["ChurnFlag"].mean() * 100
        monthly_revenue = df["MonthlyCharges"].sum()
        churned_revenue = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("Total Monthly Revenue", format_currency(monthly_revenue))
        col4.metric("Monthly Revenue at Risk", format_currency(churned_revenue))

        st.markdown("---")

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Churn by Contract")
            contract_summary = (
                df.groupby("Contract")["ChurnFlag"]
                .mean()
                .sort_values(ascending=False)
                .mul(100)
            )
            st.bar_chart(contract_summary)

        with right_col:
            st.markdown("### Churn by Internet Service")
            internet_summary = (
                df.groupby("InternetService")["ChurnFlag"]
                .mean()
                .sort_values(ascending=False)
                .mul(100)
            )
            st.bar_chart(internet_summary)

        left_col2, right_col2 = st.columns(2)

        with left_col2:
            st.markdown("### Churn by Payment Method")
            payment_summary = (
                df.groupby("PaymentMethod")["ChurnFlag"]
                .mean()
                .sort_values(ascending=False)
                .mul(100)
            )
            st.bar_chart(payment_summary)

        with right_col2:
            st.markdown("### Average Monthly Charges by Churn Status")
            monthly_charge_summary = df.groupby("Churn")["MonthlyCharges"].mean()
            st.bar_chart(monthly_charge_summary)

        st.markdown("### Key Business Takeaways")
        st.markdown(
            """
            - Month-to-month customers are the highest-risk contract segment.
            - Customers using electronic check tend to show higher churn.
            - Fiber optic customers often exhibit elevated churn relative to DSL users.
            - Customers with higher monthly charges may churn when perceived value is low.
            """
        )

    # ---------------------------
    # TAB 2: CHURN PREDICTOR
    # ---------------------------
    with tab2:
        st.subheader("Predict Churn Risk for a Customer")

        st.markdown("Enter customer details below to estimate churn probability.")

        col1, col2, col3 = st.columns(3)

        with col1:
            customerID = st.text_input("Customer ID", value="CUST-1001")
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

        with col2:
            PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with col3:
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
            PaymentMethod = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.5)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=0.5)

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([{
                "customerID": customerID,
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }])

            result = score_single_customer(model, input_df)

            st.markdown("---")
            st.markdown("### Prediction Result")

            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f'{result["churn_probability"]:.2%}')
            r2.metric("Predicted Churn", result["predicted_label"])
            r3.metric("Risk Tier", result["risk_tier"])

            if result["risk_tier"] == "High":
                st.error("This customer is at high risk of churn.")
            elif result["risk_tier"] == "Medium":
                st.warning("This customer is at moderate risk of churn.")
            else:
                st.success("This customer is at low risk of churn.")

            st.markdown("### Suggested Retention Action")

            if result["risk_tier"] == "High":
                st.write(
                    "Prioritize this customer for immediate retention outreach, "
                    "contract review, and value-added bundle offers."
                )
            elif result["risk_tier"] == "Medium":
                st.write(
                    "Consider proactive engagement, service upgrade incentives, "
                    "or onboarding support."
                )
            else:
                st.write(
                    "Maintain standard engagement and monitor for changes in risk profile."
                )

    # ---------------------------
    # TAB 3: HIGH-RISK CUSTOMERS
    # ---------------------------
    with tab3:
        st.subheader("High-Risk Customer Explorer")

        scored_df = df.copy()

        feature_cols = [col for col in scored_df.columns if col not in ["Churn", "ChurnFlag", "customerID"]]
        X = scored_df[feature_cols].copy()

        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

        scored_df["churn_probability"] = probabilities
        scored_df["predicted_churn"] = predictions
        scored_df["risk_tier"] = pd.cut(
            scored_df["churn_probability"],
            bins=[-0.01, 0.50, 0.75, 1.0],
            labels=["Low", "Medium", "High"]
        )

        risk_filter = st.selectbox("Filter by Risk Tier", ["All", "High", "Medium", "Low"])

        view_df = scored_df.copy()
        if risk_filter != "All":
            view_df = view_df[view_df["risk_tier"] == risk_filter]

        top_n = st.slider("Number of customers to display", min_value=5, max_value=50, value=20)

        display_cols = [
            "customerID", "churn_probability", "risk_tier", "MonthlyCharges",
            "tenure", "Contract", "PaymentMethod", "InternetService"
        ]

        view_df = view_df.sort_values("churn_probability", ascending=False)[display_cols].head(top_n)

        st.dataframe(view_df, use_container_width=True)

        if not view_df.empty:
            estimated_revenue_at_risk = view_df["MonthlyCharges"].sum()
            st.metric("Monthly Revenue in Displayed Segment", format_currency(estimated_revenue_at_risk))


if __name__ == "__main__":
    main()
