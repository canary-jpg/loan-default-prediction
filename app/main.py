"""
FastAPI service exposing the trained model.
Takes raw applicant field - the API caller doesn't need to know about 
engineered features (TotalTimesLate, UtilizationBucket, etc.); those are
computed internally via src/feature_engineering.py mirroring
notebooks/02_cleaning_feature_engineering.ipynb exactly

Run locally:
    uvicorn app.main:app --reload --app-dir
Then POST to /predict, e.g.:
    curl -X POST http://localhost:8000/predict-H"Content-Type:application/json" -d '{
        "RevolvingUtilizationOfUnsecuredLines":0.5,
        "age":45,
        "NumberOfTimes30_59DaysPastDueNotWorse":0,
        "DebtRatio:"0.3,
        "MonthlyIncome":5000,
        "NumberOfOpenCreditLinesAndLoans:"5,
        "NumberOfTimes90DaysLate":0,
        "NumberRealEstateLoansOrLines:"1,
        "NumberOfTime60_89DaysPastDueNotWorse:"0,
        "NumberOfDependents:"2
    }
"""

import sys
from pathlib import Path 
from typing import Optional 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 

sys.path.append(str(Path(__file__).resolve().parent.parent/"src"))
from predict import predict_from_raw 

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of serious delinquency within 2 years, "
    "given raw application fields from the Give Me Some Credit dataset schema.",
)

class BorrowerFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, description="Total balance on credit cards / credit limits"
    )
    age: int = Field(..., gt=0, description="Applicant's age in years")
    NumberOfTimes30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: Optional[float] = Field(
        None, ge=0, description="Monthly income; omit if unknown"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: Optional[int] = Field(
        None, ge=0, description="Number of dependents; omit if unknown"
    )

    class Config:
        populate_by_name = True 

@app.get('/')
def health():
    return {'status': 'ok'}


@app.post("/predict")
def predict(features: BorrowerFeatures):
    try:
        raw = {
            "RevolvingUtilizationOfUnsecuredLines": features.RevolvingUtilizationOfUnsecuredLines,
            "age": features.age,
            "NumberOfTime30-59DaysPastDueNotWorse": features.NumberOfTimes30_59DaysPastDueNotWorse,
            "DebtRatio": features.DebtRatio,
            "MonthlyIncome": features.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans": features.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": features.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": features.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": features.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": features.NumberOfDependents,
        }
        return predict_from_raw(raw)
    except FileNotFoundError:
        raise HTTPException(
        status_code=503,
        detail="Model artifact not found - train the model first (see notebooks/03)",
        )