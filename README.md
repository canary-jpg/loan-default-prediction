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
-[X] EDA
-[X] Cleaning + feature engineering
-[x] Baseline model (logistic regression)
-[x] Tree-based models (Random Forest, XGBoost/LightGBM) + comparison
-[X] Imbalance handling (class weights/SMOTE) + before/after metrics
-[X] SHAP explainability
-[X] FastAPI service
-[] Dockerized + deployed
-[] Write-up

## Results
The final XGBoost model had a ROC-AUC of 0.868, PR-AUC of 0.394, and F1 score of 0.348. SHAP shows us that `TotalTimeLate`, `RevolvingUtilizationOfUnsecuredLines`, and `age` were the features that drove loan defaults the most. These features also back credit domain intuition; for example, if an applicant has several late payments they're more likely to default on a loan.

## Limitations & next steps
### Limitations
**Data limitations**
* The dataset is anonymized with feature names documented, but there's no way to verify data provenance, collection period, or whether it reflects current lending patterns. 
* Single snapshot, no temporal dimension - the model has never been tested on how it holds up over time (e.g. a recession would likely shift the relationship between features and default risk). No macreconomic context (interest rates, unemployment) is included
* No true holdout test set - there was a train/val split created, but every modeling decision (imputation, capping thresholds, model choice) was made looking at validation metrics. A cleaner setup would reserve a separate test set touched only once, at the very end.
**Modeling limitations**
* Hyperparameters weren't extensively tuned - I used reasonable defaults/light tuning rather than a full grid/random search. So, the ceiling wasn't fully explored.
* Predicted probabilities aren't calibrated - `predict_proba` output isn't guaranteed to mean "38% chance of default" in a strict sense. If this were used for actual risk pricing (not just ranking/flagging), I would want to check calibration (e.g. a realiability diagram) first
* A PR-AUC of 0.394 means there's real room for improvement - worth stating rather than letting the ROC-AUC carry the framing, since PR-AUC is the more honest number given the imbalance.
* The threshold (0.5, giving 0.225 precision/0.776 recall) was not chosen via any cost-benefit analysis - a real deployment would need input from the business on the relative cost of a missed default vs. a wrongly flagged good borrower.
**Fairness/responsible-use**
* `age` showed up as a meaningful driver in the SHAP analysis. Age-based patterns in credit models can reflect legitimate signal (e.g. credit history length) but can also raise fair-lending concerns depending on how a model like this is actuall used. This project doesn't audit for disparate impact across demographic groups - a real deployment would need that analysis before going near a lending decision.
### Next Steps
* Hyperparameter tuning (grid search or Optuna) for a fairer ceiling comparison
* Calibration check + recalibration if probabilities need to decision-grade
* A proper held-out test set for a final, single-touch evaluation
* Monitoring plan for production (data drift, model staleness)
## Setup

```bash
python -m venv venv
source venv/bin/activate #or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Download the data from Kaggle (see `data/raw/README.md`) before running notebooks.