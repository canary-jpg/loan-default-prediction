# Loan Default Prediction

Predicting the probability that a borrower will experience serious financial distress (90+ days delinquent) within the next two years, using the Kaggle ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit/data) dataset.

**Live demo:** ["Live demo"](https://loan-default-dashboard-fkwz.onrender.com/)
**API docs:** ["API docs"](https://loan-default-prediction-fn08.onrender.com/docs)

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
-[X] Dockerized + deployed
-[X] Write-up

## Results
Final model: **XGBoost** was chosen over a logistic regression baseline and Random Forest based on PR-AUC - the more informative metric here given the dataset's 6.68% positive (default) raw, where accuracy alone would be misleading (predicting "no default" for everyone gets ~93% accuracy for free).

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.858 | 0.372 | 0.211 | 0.756 | 0.330 |
| Random Forest (class_weight) | 0.865 | 0.377 | 0.242 | 0.726 | 0.363 |
| Random Forest (SMOTE) | 0.864 | 0.386 | 0.240 | 0.730 | 0.361 |
| **XGBoost (final)** | **0.868** | **0.394** | **0.225** | **0.776** | **0.348** |

At the default 0.5 threshold, the final model achieves **0.225 precision/0.776 recall** - it catches roughly 3 in 4 actual defaulters, at the cost of a high false positive rate. That trade-off is a deliberate consequence of `scale_pos_weight`/class balancing, and the "right" threshold in a real deployment would depend on the actual business cost of a missed default vs. an unnecessarily flagged good borrower, a decision this project doesn't make on its own.
Class imbalance handling: compared class-weighting vs. SMOTE on the same Random Forest - ROC-AUC was nearly identical (0.865 vs. 0.864), while SMOTE edged out PR-AUC (0.386 vs. 0.377) at slightly lower recall. Given how close the two approaches landed, and that the final model (XGBoost) uses `scale_pos_weight` - the tree-boosting equivalent of class weighting rather than SMOTE, class weighting was the practical choice for the final pipeline: one line config vs. SMOTE's added training time complexity.

**What drives the model (SHAP analysis, notebook 04):** the top three global drivers are `TotalTimesLate`, (engineered: sum of the three "times past due" columns), `RevolovingUtilizationOfUnsecuredLines`, and `age` - all directionally consistent with domain intuition (more delinquency history and higher credit utilization both push predicted risk up). Individual predictions were explained with SHAP waterfall plots for both a high-risk (0.989 predicted probability) and low-risk (0.002) applicants, confirming the model responds sensibily in both directions rather than just pointing at the same features regardless of input.
## Limitations & next steps
### Limitations
**Data**
* The dataset is anonymized with feature names documented, but there's no way to verify data provenance, collection period, or whether it reflects current lending patterns. 
* Single snapshot, no temporal dimension - the model has never been tested on how it holds up over time (e.g. a recession would likely shift the relationship between features and default risk). No macreconomic context (interest rates, unemployment) is included
* No true holdout test set - there was a train/val split created, but every modeling decision (imputation, capping thresholds, model choice) was made looking at validation metrics. A cleaner setup would reserve a separate test set touched only once, at the very end.
**Modeling limitations**
* Hyperparameters weren't extensively tuned - I used reasonable defaults/light tuning rather than a full grid/random search. So, the ceiling wasn't fully explored.
* Predicted probabilities aren't calibrated - `predict_proba` output isn't guaranteed to mean "38% chance of default" in a strict sense. If this were used for actual risk pricing (not just ranking/flagging), I would want to check calibration (e.g. a realiability diagram) first
* A PR-AUC of 0.394 means there's real room for improvement - worth stating rather than letting the ROC-AUC carry the framing, since PR-AUC is the more honest number given the imbalance.
* The threshold (0.5, giving 0.225 precision/0.776 recall) was not chosen via any cost-benefit analysis - a real deployment would need input from the business on the relative cost of a missed default vs. a wrongly flagged good borrower.
**Engineering**
* Hit 2 dependency-pinning issues during containerization: the saved model failed to unpickle in Docker because the image installed newer scikit-learn/XGBoost versions than what the model was trained with. Fixed by pinning exact versions in `requirements.txt`. Left unpinned intially, a good reminder that model artifacts are tied to the exact library versions used to create them, not just the library itself.
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
```bash
jupyter notebook notebooks/
```
1. `01_eda.ipynb` - exploratory analysis
2 .`02_cleaning_feature_engineering.ipynb` - outputs `data/processed/{train, val}.csv`
3. `03_modeling_evaluation.ipynb` - outputs `models/model.joblib`
4. `04_explainability.ipynb` - SHAP analysis of the saved model

## runing the API locally
```bash
uvicorn app.main:app --reload --app.dir .
```
Then visit `http://localhost:8000/docs` for an interactive test UI, or:

```bash
url -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "RevolvingUtilizationOfUnsecuredLines": 0.5,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.3,
  "MonthlyIncome": 5000,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}'
```

##running the interactive demo
```bash
streamlit run app/streamlit_app.py
```

Fill in applicant details and get a live prediction plus a SHAP waterfall plot explaining that specific prediction

##running with docker
```bash
docker build -t loan-default-api -f app/DockerFile .
docker run -p 8000:8000 loan-default-api
```

##running the tests
```bash
pytest tests/
```

Download the data from Kaggle (see `data/raw/README.md`) before running notebooks.