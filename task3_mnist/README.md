# Task 3 - MNIST classifier (OOP)

`DigitClassifier` takes the name of an algorithm and classifies a 28x28x1 image. Input and
output are the same for every algorithm: a numpy array of shape `(28, 28, 1)` in, an `int`
from 0 to 9 out.

```python
import numpy as np
from digit_classifier import DigitClassifier

image = np.zeros((28, 28, 1))
DigitClassifier("cnn").predict(image)   # -> int
DigitClassifier("rf").predict(image)
DigitClassifier("rand").predict(image)
```

The goal of the task is the structure, not model quality, so the models are not trained and
`train()` raises `NotImplementedError`.

## Design

```
DigitClassificationInterface (ABC)
    preprocess(image) -> model input     abstract
    predict(x) -> int                    abstract
    train()                              raises NotImplementedError
        |
        +-- CNNModel            input: torch tensor 28x28x1
        +-- RandomForestModel   input: numpy array of 784 pixels
        +-- RandomModel         input: 10x10 center crop, returns a random digit

DigitClassifier(algorithm)
    picks a model class from MODELS by name ("cnn", "rf", "rand")
    predict(image): checks the shape is 28x28x1, then model.predict(model.preprocess(image))
```

Every model converts the 28x28x1 image to its own input format in `preprocess`, so
`DigitClassifier` does the same thing for all of them and doesn't need to know the details.

A few notes:

- `CNNModel` is a small PyTorch CNN with random weights (fixed seed). Trained weights could be
  loaded with `model.net.load_state_dict(...)`.
- scikit-learn doesn't allow `predict` on an unfitted model, so `RandomForestModel` is fitted on
  10 random images, one per digit. Its predictions are meaningless too.
- `DigitClassifier` raises `ValueError` for an unknown algorithm or an image of the wrong shape.
  Each model also checks its own input format.

### Adding a new model

1. Create a class that inherits `DigitClassificationInterface` and implements `preprocess` and
   `predict`.
2. Register it in `MODELS` in `digit_classifier/classifier.py`, e.g. `"svm": SVMModel`.

Nothing else has to change.

## Setup

Python 3.11+ (tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index installs the CPU build of PyTorch, which is enough here. Without it pip
installs the default build.

## Usage

Run from this folder.

```bash
python demo.py                  # all three algorithms on the same image
python demo.py --algorithm cnn  # only one
python -m pytest -v             # tests
```

## Files

```
digit_classifier/
    interface.py      DigitClassificationInterface
    cnn_model.py      CNNModel and the SimpleCNN network
    rf_model.py       RandomForestModel
    random_model.py   RandomModel
    classifier.py     DigitClassifier and the MODELS registry
demo.py               usage example
test_digit_classifier.py
```
