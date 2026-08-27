"""
Preprocessing pipeline for the Give Me Some Credit dataset.

I'm keeping this in a standalone module so the exact same transformations run
at training time and inference time. This is the thing that most notebook-only projects
skip and that signals production-mindedness in a portfolio piece.

Feature list below reflects the actual output of notebooks/02_cleaning_feature_engineering.ipynb 
(15 feature columns after cleaning + engineering, confirmed against data/processed/train.csv)
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

TARGET = 'SeriousDlqin2yrs'

RAW_NUMERIC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

ENGINEERED_NUMERIC_FEATURES = [
    "MonthlyIncome_was_missing",
    "TotalTimesLate",
    "IncomePerDependent",
    "HasRealEstateLoan",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = ["UtilizationBucket"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Median impute + scale numeric features; one-hot encode categoricals
        Note: by the time this pipeline runs, MonthlyIncome and 
        NumberOfDependents have already been imputed on notebook 02 (median and
        0 respectively). the SimpleImputer here is a safety net for the val/inference-time
        path, not the primary imputation strategy.
     """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ]
     )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("one-hot", OneHotEncoder(handle_unknown='ignore')),
        ]
     )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor 

def build_full_pipeline(model) -> Pipeline:
    """Wrap a preprocessing step + estimator into a single Pipeline so
    train.py and predict.py never touch raw features differently. """
    preprocessor = build_preprocessing_pipeline()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

#for logistic regression baseline specifically
LOGISTIC_REGRESSION_NUMERIC_FEATURES = [
    f for f in NUMERIC_FEATURES 
    if f 
    not in ("NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate")
]

