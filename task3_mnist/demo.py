"""Run DigitClassifier with every algorithm on the same image.

Usage:
    python demo.py
    python demo.py --algorithm rf
"""

import argparse

import numpy as np

from digit_classifier import MODELS, DigitClassifier


def make_image() -> np.ndarray:
    """A rough "1": a vertical stroke in the middle of a black 28x28x1 image."""
    image = np.zeros((28, 28, 1), dtype=np.float32)
    image[4:24, 13:15, 0] = 1.0
    return image


def main():
    parser = argparse.ArgumentParser(description="DigitClassifier demo.")
    parser.add_argument("--algorithm", choices=list(MODELS), help="run only this algorithm")
    args = parser.parse_args()

    image = make_image()
    algorithms = [args.algorithm] if args.algorithm else list(MODELS)

    print(f"Input image shape: {image.shape}")
    for name in algorithms:
        prediction = DigitClassifier(name).predict(image)
        print(f"{name:>4}: {prediction}")

    print("Models are not trained, so the predictions are meaningless.")


if __name__ == "__main__":
    main()
