"""Random Forest digit classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .interface import DigitClassificationInterface

N_PIXELS = 28 * 28


class RandomForestModel(DigitClassificationInterface):
    """Random Forest classifier. Input: 1-d array of 784 pixels."""

    def __init__(self, seed: int = 42):
        # sklearn can't predict with an unfitted model, so we fit it on
        # 10 random images (one per digit). Predictions are meaningless,
        # a real model would be loaded from disk instead.
        rng = np.random.default_rng(seed)
        self.forest = RandomForestClassifier(n_estimators=10, random_state=seed)
        self.forest.fit(rng.random((10, N_PIXELS)), np.arange(10))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # 28x28x1 -> 784
        return image.reshape(N_PIXELS)

    def predict(self, x: np.ndarray) -> int:
        if x.shape != (N_PIXELS,):
            raise ValueError(f"Expected a 1-d array of {N_PIXELS} pixels, got {x.shape}")
        # sklearn expects a batch of samples, so pass one row
        return int(self.forest.predict(x.reshape(1, -1))[0])
