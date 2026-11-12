"""Проверки полносвязной нейросети."""
import numpy as np
import pandas as pd
import pytest

from src.models import build_model
from src.neural import MLPRegressor


@pytest.fixture
def linear_data():
    """Линейная зависимость с целевой переменной в диапазоне значений PSNR."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x": rng.uniform(-1, 1, 400), "z": rng.uniform(-1, 1, 400)})
    return features, 28 + 2 * features["x"] - features["z"]


def test_mlp_learns_function_on_psnr_scale(linear_data):
    features, target = linear_data
    model = MLPRegressor(hidden_units=[32], epochs=150, batch_size=32, loss="mse").fit(features, target)
    assert np.abs(model.predict(features) - target).mean() < 0.15


@pytest.mark.parametrize("options", [
    {"hidden_units": [16, 8], "batch_norm": True, "dropout": 0.2, "activation": "gelu"},
    {"optimizer": "sgd", "scheduler": "cosine", "loss": "huber"},
    {"optimizer": "adamw"},
    {"optimizer": "rmsprop"},
])
def test_mlp_options_build_and_predict(linear_data, options):
    features, target = linear_data
    spec = {"type": "mlp", "clip": [0.001, 0.999], "scaler": "standard", "params": {"epochs": 2, **options}}
    predicted = build_model(spec).fit(features, target).predict(features)
    assert predicted.shape == (len(features),) and np.isfinite(predicted).all()


@pytest.mark.parametrize("options", [{"scheduler": "step"}, {"optimizer": "lion"}])
def test_mlp_rejects_unknown_options(linear_data, options):
    features, target = linear_data
    with pytest.raises(ValueError):
        MLPRegressor(epochs=1, **options).fit(features, target)
