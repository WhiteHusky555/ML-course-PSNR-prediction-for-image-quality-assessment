"""Проверки подбора гиперпараметров."""
import numpy as np
import optuna
import pandas as pd

from src.tuning import suggest_params, tune_model
from src.validation import make_folds


def test_suggest_params_follows_space_types():
    space = {"depth": {"low": 2, "high": 8}, "rate": {"low": 0.01, "high": 1, "log": True},
             "loss": {"choices": ["l1", "l2"]}}
    trial = optuna.trial.FixedTrial({"depth": 4, "rate": 0.1, "loss": "l1"})
    assert suggest_params(trial, space) == {"depth": 4, "rate": 0.1, "loss": "l1"}


def test_tune_model_improves_on_bad_start():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    train["PSNR"] = train["a"] * 3 - train["b"] + rng.normal(scale=0.1, size=200)
    train["source"] = np.arange(200) // 5
    train["variant"] = np.arange(200) % 5
    folds = make_folds(train, "PSNR", 3, 5, 0)
    spec = {"type": "ridge", "params": {"alpha": 10000.0}, "search": {"alpha": {"low": 0.001, "high": 10000.0, "log": True}}}

    study = tune_model(spec, train, ["a", "b", "c"], folds, "PSNR", {"n_trials": 10}, seed=0)

    assert study.trials[0].params == {"alpha": 10000.0}
    assert study.best_value < study.trials[0].value / 2
    assert study.best_params["alpha"] < 10000.0
