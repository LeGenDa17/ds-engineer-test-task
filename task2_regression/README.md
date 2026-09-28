# Task 2 - Regression on tabular data

Predict `target` from 53 anonymized features. Metric: RMSE.

## Result

EDA showed that the target is a function of two features:

```
target = feature_6² + feature_7
```

The other 51 features are noise. The model is a linear regression on `[feature_6², feature_7]`.
It learns coefficients 1 and 1 with intercept 0, and its validation RMSE is practically 0
(float rounding only).

Validation RMSE of the models I compared (80/20 split of `train.csv`):

| Model | RMSE |
|---|---|
| Always predict the mean | 29.0 |
| Linear regression, 53 raw features | 29.0 |
| Random Forest, 53 raw features | 0.0038 |
| Linear regression, [6², 7] | ~0 |

How I got there is in [eda.ipynb](eda.ipynb).

## Setup

Python 3.11+ (tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run from this folder.

Train (reads `data/train.csv`, saves `model.joblib`):

```bash
python train.py
```

Predict (reads `data/hidden_test.csv` and `model.joblib`, saves `predictions.csv`):

```bash
python predict.py
```

Both scripts accept paths as arguments, see `python train.py --help` and `python predict.py --help`.

## Files

```
eda.ipynb          exploratory data analysis and model comparison
features.py        feature engineering shared by train.py and predict.py
train.py           trains the model and saves it
predict.py         loads the model and makes predictions
predictions.csv    predictions for hidden_test.csv, one row per input row, same order
model.joblib       trained model
data/              train.csv, hidden_test.csv
```
