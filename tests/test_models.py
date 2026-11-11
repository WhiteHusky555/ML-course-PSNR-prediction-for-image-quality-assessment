"""Проверки создания моделей и развёртывания сеток параметров."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
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


@pytest.mark.parametrize("spec", [
    {"type": "knn", "scaler": "standard", "params": {"n_neighbors": 3, "weights": "distance"}},
    {"type": "tree", "params": {"max_depth": None, "random_state": 0}},
    {"type": "forest", "params": {"n_estimators": 10, "random_state": 0}},
])
def test_nonlinear_models_fit_step_function(spec):
    features = np.linspace(0, 1, 200).reshape(-1, 1)
    target = (features[:, 0] > 0.5).astype(float)
    predicted = build_model(spec).fit(features, target).predict([[0.1], [0.9]])
    assert predicted == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize("spec", [
    {"type": "lightgbm", "params": {"n_estimators": 100, "objective": "l1", "verbose": -1}},
    {"type": "xgboost", "params": {"n_estimators": 100, "objective": "reg:absoluteerror"}},
    {"type": "catboost", "params": {"iterations": 100, "loss_function": "MAE", "verbose": 0,
                                    "allow_writing_files": False}},
    {"type": "gradboost", "params": {"n_estimators": 100, "loss": "absolute_error"}},
    {"type": "histgb", "params": {"max_iter": 100, "loss": "absolute_error", "min_samples_leaf": 5}},
])
def test_boosting_models_fit_step_function(spec):
    features = pd.DataFrame({"x": np.linspace(0, 1, 200)})
    target = (features["x"] > 0.5).astype(float)
    predicted = build_model(spec).fit(features, target).predict(pd.DataFrame({"x": [0.1, 0.9]}))
    assert predicted == pytest.approx([0.0, 1.0], abs=0.05)


@pytest.mark.parametrize("spec", [
    {"type": "xgboost", "params": {"n_estimators": 100, "objective": "reg:pseudohubererror"}},
    {"type": "catboost", "params": {"iterations": 100, "loss_function": "Huber:delta=1.0", "verbose": 0,
                                    "allow_writing_files": False}},
])
def test_centered_target_lets_huber_reach_large_values(spec):
    features = np.linspace(0, 1, 200).reshape(-1, 1)
    target = 28 + (features[:, 0] > 0.5)
    centered = {**spec, "center_target": True}
    predicted = build_model(centered).fit(features, target).predict(np.array([[0.1], [0.9]]))
    assert predicted == pytest.approx([28.0, 29.0], abs=0.1)
    assert abs(build_model(spec).fit(features, target).predict(np.array([[0.1]]))[0] - 28) > 0.5


@pytest.mark.parametrize(("spec", "message"), [
    ({"type": "svm"}, "Неизвестный тип модели"),
    ({"type": "ridge", "scaler": "minmax"}, "Неизвестный способ масштабирования"),
])
def test_build_model_rejects_unknown_names(spec, message):
    with pytest.raises(ValueError, match=message):
        build_model(spec)


@pytest.mark.parametrize("path", ["config.yaml", *sorted(str(p) for p in Path("experiments").glob("*.yaml"))])
def test_every_configured_model_can_be_built(path):
    models = expand_grid(load_config(path)["models"])
    assert models
    for spec in models.values():
        build_model(spec)
