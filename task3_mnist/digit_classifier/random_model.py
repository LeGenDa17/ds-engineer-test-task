"""Model that returns a random digit."""

import numpy as np

from .interface import DigitClassificationInterface

CROP_SIZE = 10


class RandomModel(DigitClassificationInterface):
    """Returns a random digit. Input: 10x10 center crop of the image."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # 28x28x1 -> 10x10: cut out the center and drop the channel axis
        start = (image.shape[0] - CROP_SIZE) // 2
        return image[start:start + CROP_SIZE, start:start + CROP_SIZE, 0]

    def predict(self, x: np.ndarray) -> int:
        if x.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(f"Expected a {CROP_SIZE}x{CROP_SIZE} array, got {x.shape}")
        return int(self.rng.integers(0, 10))
