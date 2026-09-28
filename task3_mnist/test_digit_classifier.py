"""Tests for the digit_classifier package. Run from the task3_mnist folder: python -m pytest -v"""

import numpy as np
import pytest
import torch

from digit_classifier import (
    MODELS,
    CNNModel,
    DigitClassificationInterface,
    DigitClassifier,
    RandomForestModel,
    RandomModel,
)

ALGORITHMS = ["cnn", "rf", "rand"]


@pytest.fixture
def image():
    return np.random.default_rng(0).random((28, 28, 1))


# --- DigitClassifier: same input and output for every algorithm --------------

@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_predict_returns_a_digit(algorithm, image):
    prediction = DigitClassifier(algorithm).predict(image)
    assert type(prediction) is int
    assert 0 <= prediction <= 9


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_predict_accepts_nested_lists(algorithm, image):
    assert 0 <= DigitClassifier(algorithm).predict(image.tolist()) <= 9


@pytest.mark.parametrize("shape", [(28, 28), (784,), (28, 28, 3), (10, 10)])
def test_wrong_image_shape_raises(shape):
    with pytest.raises(ValueError):
        DigitClassifier("cnn").predict(np.zeros(shape))


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="cnn, rf, rand"):
        DigitClassifier("svm")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_is_not_implemented(algorithm):
    with pytest.raises(NotImplementedError):
        DigitClassifier(algorithm).train()
    with pytest.raises(NotImplementedError):
        MODELS[algorithm]().train()


# --- Each model gets its own input format -------------------------------------

def test_cnn_input_is_tensor_28x28x1(image):
    x = CNNModel().preprocess(image)
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (28, 28, 1)


def test_rf_input_is_flat_784(image):
    x = RandomForestModel().preprocess(image)
    assert x.shape == (784,)
    assert np.array_equal(x, image.flatten())


def test_random_input_is_center_crop_10x10():
    image = np.zeros((28, 28, 1))
    image[9:19, 9:19, 0] = 1  # the center 10x10 square is white
    crop = RandomModel().preprocess(image)
    assert crop.shape == (10, 10)
    assert crop.min() == 1


@pytest.mark.parametrize(
    "model, bad_input",
    [
        (CNNModel(), torch.zeros(1, 28, 28)),
        (RandomForestModel(), np.zeros((28, 28))),
        (RandomModel(), np.zeros((28, 28))),
    ],
)
def test_model_rejects_wrong_input_format(model, bad_input):
    with pytest.raises(ValueError):
        model.predict(bad_input)


def test_random_model_is_reproducible_with_seed():
    crop = np.zeros((10, 10))
    first = [RandomModel(seed=1).predict(crop) for _ in range(5)]
    second = [RandomModel(seed=1).predict(crop) for _ in range(5)]
    assert first == second


# --- Interface ----------------------------------------------------------------

def test_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DigitClassificationInterface()


def test_model_without_predict_cannot_be_created():
    class BadModel(DigitClassificationInterface):
        def preprocess(self, image):
            return image

    with pytest.raises(TypeError):
        BadModel()


def test_new_model_can_be_registered(monkeypatch, image):
    class AlwaysSeven(DigitClassificationInterface):
        def preprocess(self, image):
            return image

        def predict(self, x):
            return 7

    monkeypatch.setitem(MODELS, "seven", AlwaysSeven)
    assert DigitClassifier("seven").predict(image) == 7
