"""
Loads the trained pipeline and runs predictions on new data.
Shared by app/main.py so training and serving never drift apart
"""

import joblib 
import pandas as pd 

from pipeline import ALL_FEATURES

_MODEL = None 

def get_model(model_path:str = "models/model.joblib") -> dict:
    """features: dict matching ALL_FEATURES keys (raw + engineered columms,
    including UtilizationBucket as one of "low/medium/high") """
    model = get_model(model_path)
    row = pd.DataFrame({col: features.get(col) for col in ALL_FEATURES})
    proba = model.predict_proba(row)[0,1]
    return {"default_proability": float(proba), "predicted_class": int(proba >= 0.5)}