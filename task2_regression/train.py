"""Train a linear regression on the engineered features and save it to disk.

Usage:
    python train.py
    python train.py --data data/train.csv --model model.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, make_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train the regression model.")
    parser.add_argument("--data", default="data/train.csv", help="path to the training CSV")
    parser.add_argument("--model", default="model.joblib", help="where to save the trained model")
    parser.add_argument("--val-size", type=float, default=0.2, help="share of data used for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the train/validation split")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    X = make_features(df)
    y = df["target"]
    print(f"Loaded {len(df)} rows from {args.data}")

    # 1. Check the model on data it hasn't seen.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed
    )
    model = LinearRegression().fit(X_train, y_train)
    val_rmse = root_mean_squared_error(y_val, model.predict(X_val))
    print(f"Validation RMSE: {val_rmse:.6f}")

    # 2. Retrain on all data for the final model.
    model = LinearRegression().fit(X, y)
    coefs = ", ".join(f"{name} = {c:.6f}" for name, c in zip(FEATURE_NAMES, model.coef_))
    print(f"Coefficients: {coefs}, intercept = {model.intercept_:.6f}")

    joblib.dump(model, args.model)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
