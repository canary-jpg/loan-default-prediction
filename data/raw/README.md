# Raw data

Not committed to the repo (see `.gitignore`).

Download from Kaggle:
https://www.kaggle.com/c/GiveMeSomeCredit/data

Or via the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c GiveMeSomeCredit -p data/raw
unzip data/raw/GiveMeSomeCredit.zip -d data/raw
```

Expected files:
- `cs-training.csv`
- `cs-test.csv`
- `Data Dictionary.xls`
