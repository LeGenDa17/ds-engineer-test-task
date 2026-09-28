"""Predict the target for new data with a model trained by train.py.

Usage:
    python predict.py
    python predict.py --data data/hidden_test.csv --model model.joblib --output predictions.csv
"""

import argparse

import joblib
import pandas as pd

from features import make_features


def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions with the trained model.")
    parser.add_argument("--data", default="data/hidden_test.csv", help="path to the CSV with features")
    parser.add_argument("--model", default="model.joblib", help="path to the trained model")
    parser.add_argument("--output", default="predictions.csv", help="where to save the predictions")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    model = joblib.load(args.model)
    print(f"Loaded {len(df)} rows from {args.data} and model from {args.model}")

    predictions = model.predict(make_features(df))

    # One prediction per input row, in the same order as the input file.
    pd.DataFrame({"target": predictions}).to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
