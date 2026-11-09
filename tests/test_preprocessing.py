"""Проверки шагов предобработки и их использования в моделях."""
import numpy as np
import pandas as pd
import pytest

from src.models import build_model, model_columns
from src.preprocessing import CorrelationFilter, QuantileClipper


def test_clipper_uses_bounds_learned_on_training_data():
    train = np.arange(101, dtype=float).reshape(-1, 1)
    clipper = QuantileClipper(0.1, 0.9).fit(train)

    assert clipper.transform([[-50.0], [50.0], [500.0]]).ravel().tolist() == [10.0, 50.0, 90.0]


def test_correlation_filter_drops_monotone_copies_and_constants():
    rng = np.random.default_rng(0)
    base, other = rng.normal(size=(2, 300))
    features = np.column_stack([base, np.exp(base), other, np.full(300, 3.0), 0.5 * other + base])

    selector = CorrelationFilter(0.95).fit(features)

    assert selector.keep_.tolist() == [0, 2, 4]
    assert selector.transform(features).shape == (300, 3)


def test_preprocessing_steps_go_before_model_in_given_order():
    spec = {"type": "ridge", "clip": [0.01, 0.99], "select": 0.9, "scaler": "quantile", "pca": 2}
    steps = [type(step).__name__ for step in build_model(spec)]
    assert steps == ["QuantileClipper", "CorrelationFilter", "QuantileTransformer", "PCA", "Ridge"]


@pytest.mark.parametrize("encoding", ["onehot", "ordinal", "target"])
def test_variant_encoding_lets_model_use_variant(encoding):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"blur": rng.normal(size=200), "variant": np.tile(["1", "2"], 100)})
    target = np.where(frame["variant"] == "2", 10.0, 0.0) + frame["blur"]
    spec = {"type": "linear", "scaler": "standard", "variant": encoding}
    columns = model_columns(spec, ["blur"])

    predictions = build_model(spec).fit(frame[columns], target).predict(frame[columns])

    assert columns == ["blur", "variant"]
    assert np.abs(predictions - target).mean() < 0.5


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="Неизвестный способ кодирования"):
        build_model({"type": "ridge", "variant": "hashing"})
