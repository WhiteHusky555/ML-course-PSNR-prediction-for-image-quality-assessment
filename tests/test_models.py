"""Проверки создания моделей и развёртывания сеток параметров."""
import numpy as np
import pytest

from src.models import build_model, expand_grid


def test_expand_grid_creates_model_per_combination():
    models = expand_grid({
        "median": {"type": "dummy", "params": {"strategy": "median"}},
        "net": {
            "type": "elasticnet",
            "scaler": "standard",
            "params": {"alpha": 5.0, "max_iter": 100},
            "grid": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]},
        },
    })

    assert list(models) == [
        "median",
        "net alpha=0.1 l1_ratio=0.2", "net alpha=0.1 l1_ratio=0.8",
        "net alpha=1.0 l1_ratio=0.2", "net alpha=1.0 l1_ratio=0.8",
    ]
    assert models["median"] == {"type": "dummy", "params": {"strategy": "median"}}
    assert models["net alpha=1.0 l1_ratio=0.8"] == {
        "type": "elasticnet",
        "scaler": "standard",
        "params": {"alpha": 1.0, "max_iter": 100, "l1_ratio": 0.8},
    }


def test_scaled_model_does_not_depend_on_feature_units():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    target = features @ [1.0, 2.0, 3.0] + rng.normal(scale=0.1, size=200)
    rescaled = features * [1.0, 1e4, 1e-3]
    spec = {"type": "ridge", "params": {"alpha": 10.0}}

    scaled = {**spec, "scaler": "standard"}
    assert build_model(scaled).fit(features, target).predict(features) == pytest.approx(
        build_model(scaled).fit(rescaled, target).predict(rescaled))
    assert build_model(spec).fit(features, target).predict(features) != pytest.approx(
        build_model(spec).fit(rescaled, target).predict(rescaled))


@pytest.mark.parametrize(("spec", "message"), [
    ({"type": "svm"}, "Неизвестный тип модели"),
    ({"type": "ridge", "scaler": "minmax"}, "Неизвестный способ масштабирования"),
])
def test_build_model_rejects_unknown_names(spec, message):
    with pytest.raises(ValueError, match=message):
        build_model(spec)
