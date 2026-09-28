"""DigitClassifier: one entry point for all digit models."""

import numpy as np

from .cnn_model import CNNModel
from .interface import DigitClassificationInterface
from .random_model import RandomModel
from .rf_model import RandomForestModel

IMAGE_SHAPE = (28, 28, 1)

# Algorithm name -> model class. To add a new model, implement
# DigitClassificationInterface and register it here.
MODELS: dict[str, type[DigitClassificationInterface]] = {
    "cnn": CNNModel,
    "rf": RandomForestModel,
    "rand": RandomModel,
}


class DigitClassifier:
    """Classifies a 28x28x1 image with the chosen algorithm: cnn, rf or rand.

    Input and output are the same for every algorithm, each model converts
    the image to its own format in ``preprocess``.
    """

    def __init__(self, algorithm: str):
        if algorithm not in MODELS:
            raise ValueError(f"Unknown algorithm '{algorithm}', choose one of: {', '.join(MODELS)}")
        self.algorithm = algorithm
        self.model = MODELS[algorithm]()

    def predict(self, image: np.ndarray) -> int:
        """Return the predicted digit (0-9) for a 28x28x1 image."""
        image = np.asarray(image)
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"Expected an image of shape {IMAGE_SHAPE}, got {image.shape}")
        return self.model.predict(self.model.preprocess(image))

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented")
