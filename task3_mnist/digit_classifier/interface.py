"""Common interface for all digit classification models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DigitClassificationInterface(ABC):
    """Contract every digit model has to follow.

    A model gets a 28x28x1 image, converts it to its own input format in
    ``preprocess`` and returns a digit (0-9) from ``predict``.
    """

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Convert a 28x28x1 image to the input format of this model."""

    @abstractmethod
    def predict(self, x: Any) -> int:
        """Return the predicted digit for an input produced by ``preprocess``."""

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented")
