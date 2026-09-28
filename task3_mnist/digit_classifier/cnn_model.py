"""Convolutional neural network digit classifier."""

import numpy as np
import torch
from torch import nn

from .interface import DigitClassificationInterface

IMAGE_SHAPE = (28, 28, 1)


class SimpleCNN(nn.Module):
    """Two conv blocks and a linear layer, 1x28x28 -> 10 class scores."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x7x7
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(start_dim=1))


class CNNModel(DigitClassificationInterface):
    """CNN classifier. Input: tensor 28x28x1.

    The weights are random (not trained), so predictions are meaningless.
    Trained weights could be loaded with ``self.net.load_state_dict(...)``.
    """

    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)
        self.net = SimpleCNN().eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        # numpy 28x28x1 -> torch tensor 28x28x1
        return torch.as_tensor(image, dtype=torch.float32)

    def predict(self, x: torch.Tensor) -> int:
        if tuple(x.shape) != IMAGE_SHAPE:
            raise ValueError(f"Expected a tensor of shape {IMAGE_SHAPE}, got {tuple(x.shape)}")
        # PyTorch wants (batch, channels, height, width): 28x28x1 -> 1x1x28x28
        batch = x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            scores = self.net(batch)
        return int(scores.argmax(dim=1).item())
