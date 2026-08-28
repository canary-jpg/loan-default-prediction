"""
Transforms raw applicant input into the engineered feature set the model
was trained on. Mirrors the logic in notebooks/02_cleaning_feature_engineering.ipynb exactly - if
that notebook's cleaning/engineering steps change, update this file to match, or predictions 
with silently drift from what the model actually learned.
"""

from pathlib import Path 
import pandas as pd 

_TRAIN_PATH = Path(__file__).resolve().parent.parent/"data"/"processed"/"train.csv"

#caps applied during cleaning notebook (notebook 02, confirmed against the actual EDA/cleaning output)
#these are fixed data-cleaning decisions, not something learned from the data, so they're hardcoded here
#to match exactly what produced the training set the current model was fit on
LATE_COLUMN_CAP = 15
UTILIZATION_CAP = 1.37 
DEBT_RATIO_CAP = 6186.02 

LATE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

def _load_montly_income_median() -> float:
    """Pulls the median from the actual processed training data rather
    than hardcoding a number, so this stays correct if the model is retrained """
    try:
        train_df = pd.read_csv("_TRAIN_PATH")
        return float(train_df['MonthlyIncome'].median())
    except FileNotFoundError:
        #fallback so this module can still be imported (e.g. in tests) before
        #before data/processed/train.csv exists - not used once the real pipeline is in place
        return 5400.0 

MONTHLY_INCOME_MEDIAN = _load_montly_income_median()

def engineer_features(raw:dict) -> dict:
    """
    raw: dict with the original Give Me Some Credit fields.
    `MonthlyIncome` and `NumberOfDependents` may be None (unknown at
    application time) - everything else is required

    Returns: dict with all 15 model feature columns (cleaned raw +
    engineered), ready to pass into the trained pipeline via predict_one()
    """

    features = dict(raw)
    #missing value handing (mirrors notebook 02, section 1)
    monthly_income = features.get("MonthlyIncome")
    was_missing = monthly_income is None
    if was_missing:
        monthly_income = MONTHLY_INCOME_MEDIAN
    features['MonthlyIncome'] = monthly_income 
    features['MonthlyIncome_was_missing'] = int(was_missing)

    num_dependents = features.get("NumberOfDependents")
    if num_dependents is None:
        num_dependents = 0
    features['NumberOfDependents'] = num_dependents

    #capping (mirrors notebook 02, sections 2-3)
    for col in LATE_COLUMNS:
        features[col] = min(features[col], LATE_COLUMN_CAP)
    features['RevolvingUtilizationOfUnsecuredLines'] = min(
        features['RevolvingUtilizationOfUnsecuredLines'], UTILIZATION_CAP
    )
    features['DebtRatio'] = min(features['DebtRatio'], DEBT_RATIO_CAP)

    #feature engineering (mirrors notebook 02, section 4)
    features["TotalTimesLate"] = sum(features[c] for c in LATE_COLUMNS)
    features['IncomePerDependent'] = monthly_income/(num_dependents + 1)
    features['HasRealEstateLoan'] = int(features['NumberRealEstateLoansOrLines'] > 0)

    utilization = features['RevolvingUtilizationOfUnsecuredLines']
    if utilization <= 0.3:
        bucket = 'low'
    elif utilization <= 0.7:
        bucket = 'medium'
    else:
        bucket = 'high'
    features['UtilizationBucket'] = bucket 

    return features 
    
        