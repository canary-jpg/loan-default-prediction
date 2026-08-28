"""
Streamlit front-end for the loan default model.
Lets a non-technical reviewer type in applicant values and see a live
prediction, plus a SHAP-based explanation of what drove that specific
prediction

Run locally:
    streamlit run app/streamlit_app.py
"""

import sys 
from pathlib import Path 
import joblib 
import matplotlib.pyplot as plt 
import pandas as pd 
import shap 
import streamlit as st 

sys.path.append(str(Path(__file__).resolve().parent.parent/"src"))
from feature_engineering import engineer_features 
from pipeline import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES 

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent.parent/"models"/"model.joblib"

@st.cache_resource 
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer(_pipeline):
    """SHAP TreeExplainer needs the raw model, not the full sklearn Pipeline -
    same approach as notebooks/04_explainability.ipynb """
    model = _pipeline.named_steps["model"]
    return shap.TreeExplainer(model)

def get_feature_names(preprocessor):
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["one-hot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return list(NUMERIC_FEATURES) + list(cat_feature_names)

st.title("💳 Loan Default Risk Predictor")
st.caption(
    "Predicts the probability of serious delinquency (90+ days) within two years,"
    "using the Give Me Some Credit dataset. [View source on GitHub](#)"
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model at `models/model.joblib`. "
        "Run `notebooks/03_modeling_evaluation.ipynb` first to train and save one."
    )
    st.stop()

pipeline = load_pipeline()
preprocessor = pipeline.named_steps["preprocessor"]
explainer = load_explainer(pipeline)

st.subheader("Applicant details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    revolving_util = st.slider(
        "Revolving utilization of unsecured lines",
        min_value=0.0, max_value=2.0, value=0.3, step=0.01,
        help="Total balance on credit cards / credit limits Values above 1.0 "
        "are capped at 1.37 during preprocessing, matching training-time cleaning.",
    )
    debt_ratio = st.number_input(
        "Debt Ratio", min_value=0.0, value=0.3, step=0.01,
        help="Monthly debt payments/monthly gross income",
    )
    open_credit_lines = st.number_input(
        "Number of open credit lines and loans", min_value=0, value=5
    )
    real_estate_loans = st.number_input(
        "Number of real estate loans/lines", min_value=0, value=1
    )

with col2:
    income_unknown = st.checkbox("Monthly income unknown")
    monthly_income = (
        None if income_unknown
        else st.number_input("Monthly income ($)", min_value=0, value=5000, step=100)
    )
    dependents_unknown = st.checkbox("Number of dependents unknown")
    num_dependents = (
        None if dependents_unknown 
        else st.number_input("Number of dependents", min_value=0, value=0)
    )
    late_30_59 = st.number_input("Times 30-59 days past due", min_value=0, value=0)
    late_60_89 = st.number_input("Times 60-89 days past due", min_value=0, value=0)
    late_90 = st.number_input("Times 90+ days late", min_value=0, value=0)

if st.button("Predict default risk", type="primary"):
    raw = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": num_dependents,
    }

    engineered = engineer_features(raw)
    row = pd.DataFrame([{col: engineered.get(col) for col in ALL_FEATURES}])
    proba = pipeline.predict_proba(row)[0, 1]

    st.divider()
    st.subheader("Result")

    risk_col, gauge_col = st.columns([1, 2])
    with risk_col:
        st.metric("Default probability", f"{proba:.1%}")
        risk_label = "High risk" if proba >= 0.5 else "Low risk"
        st.markdown(f"**Classification:** {risk_label}")

    st.divider()
    st.subheader("Why this prediction - top contributing factors")
    st.caption(
        "SHAP values show how much each feature pushed this specific "
        "prediction up (toward default) or down (toward no default)."
    )

    feature_names = get_feature_names(preprocessor)
    row_transformed = preprocessor.transform(row)
    shap_values = explainer(pd.DataFrame(row_transformed, columns=feature_names))

    fig, ax = plt.subplots(figsize=(8,4))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    st.pyplot(fig)

st.divider()
st.caption(
    "⚠️ This is a portfolio demo trained on an anonymized public dataset. "
    "It is not calibrated or fairness-audited and should not be used for "
    "actual lending decisions."
)