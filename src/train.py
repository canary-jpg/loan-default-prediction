"""
Trains the final model and saves it to models/model.joblib
Reads from data/processed/{train, val}.csv produced by
notebooks/02_cleaning_feature_engineering.ipynb - run that notebook first.
Usage:
    python src/train.py
"""

import argparse
import joblib 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from pipeline import ALL_FEATURES, TARGET, build_full_pipeline

#swap in whichever model wins the comparison in notebook 03

def load_processed(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def main(train_path: str, val_path: str, model_out: str):
    train_df = load_processed(train_path)
    val_df = load_processed(val_path)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced", #placeholder: compare against SMOTE in notebook 03
        random_state=42,
        n_jobs=-1,
    )

    pipeline = build_full_pipeline(model)
    pipeline.fit(X_train, y_train)

    #recall/precision/F1 matter more than accuracy b/c large class imbalance
    #positive rate confirmed in EDA, accuracy alone would be misleading
    val_proba = pipeline.predict_proba(X_val)[:,1]
    val_pred = pipeline.predict(X_val)

    print("ROC-AUC:", round(roc_auc_score(y_val, val_proba), 4))
    print("PR-AUC:", round(average_precision_score(y_val, val_proba)))
    print(classification_report(y_val, val_pred, digits=3))

    joblib.dump(pipeline, model_out)
    print(f"Saved trained pipeline to {model_out}")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--model-out", default="data/processed/model.joblib")
    args = parser.parse_args()

    main(args.train, args.val, args.model_out)
