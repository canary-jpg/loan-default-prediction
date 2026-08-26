# Loan Default Prediction

Predicting the probability that a borrower will experience serious financial distress (90+ days delinquent) within the next two years, using the Kaggle ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit/data) dataset.

## Business framing

Lenders lose money two ways: rejecting good borrowers (lost revenue) and approving borrowers who default (lost principal). A well-calibrated risk model lets a lender set a threshold that balances these costs instead of guessing. This project builds the model end-to-end and serves it as an API that could sit behind a loan application flow.

## Project Structure

```
loan-default-prediction/
├── README.md
├── data/
│   ├── raw/            # original CSV(s), not committed — see data/raw/README.md
│   └── processed/       # cleaned/engineered data, not committed
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning_feature_engineering.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── pipeline.py      # sklearn ColumnTransformer / Pipeline definitions
│   ├── train.py         # trains + saves the final model
│   └── predict.py       # loads model, runs inference
├── app/
│   ├── main.py           # FastAPI app exposing /predict
│   └── Dockerfile
├── models/                # saved model artifacts, not committed
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── .gitignore
```

## Status
-[] EDA
-[] Cleaning + feature engineering
-[] Baseline model (logistic regression)
-[] Tree-based models (Random Forest, XGBoost/LightGBM) + comparison
-[] Imbalance handling (class weights/SMOTE) + before/after metrics
-[] SHAP explainability
-[] FastAPI service
-[] Dockerized + deployed
-[] Write-up

## Results

## Limitations & next steps

## Setup

```bash
python -m venv venv
source venv/bin/activate #or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Download the data from Kaggle (see `data/raw/README.md`) before running notebooks.