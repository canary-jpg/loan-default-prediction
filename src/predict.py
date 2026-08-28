"""
Loads the trained pipeline and runs predictions on new data.
Shared by app/main.py so training and serving never drift apart
"""

import joblib 
import pandas as pd 
from feature_engineering import engineer_features
from pipeline import ALL_FEATURES

_MODEL = None 

def get_model(model_path:str = "models/model.joblib") -> dict:
   global _MODEL
   if _MODEL is None: 
    _MODEL = joblib.load(model_path)
   return _MODEL

def predict_one(features: dict, model_path:str = 'models/model.joblib') -> dict:
    """features: dict already matching ALL_FEATURES keys (raw + engineered) 
    columns, including UtilizationBucket as one of "low"/"medium"/"high"
    Use predict_from_raw() instead if you only have raw applicant field """
    model = get_model(model_path)
    row = pd.DataFrame([{col: features.get(col) for col in ALL_FEATURES}])
    proba = model.predict_proba(row)[0,1]
    return {"default_probability": float(proba), "predicted_class": int(proba >= 0.5)}

def predict_from_raw(raw_features: dict, model_path:str = "models/model.joblib") -> dict:
    """raw_features: dict with the 10 original applicatnt fields
    (MonthlyIncome and NumberOfDependents may be None). Computes the
    engineered features (mirroring notebook 02) before predicting, so
    callers never need to konw about TotalTimesLate, UtilizationBucket,
    etc. this is what app/main.py should call """
    engineered = engineer_features(raw_features)
    return predict_one(engineered, model_path)