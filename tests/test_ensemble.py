"""Проверки ансамбля из прогнозов базовых моделей."""
import numpy as np
import pandas as pd
import pytest

from main import run
from src.ensemble import PREDICTION_PREFIX, WeightedBlend, read_predictions
from src.models import model_columns


def test_blend_weights_choose_exact_model():
    rng = np.random.default_rng(0)
    target = rng.normal(30, 2, 200)
    predictions = np.column_stack([target + rng.normal(0, 1, 200), target, target - 3])
    weights = WeightedBlend().fit(predictions, target).weights_
    assert np.allclose(weights, [0, 1, 0], atol=1e-6)


def test_blend_weights_are_convex_and_beat_each_model():
    rng = np.random.default_rng(1)
    target = rng.normal(30, 2, 300)
    predictions = np.column_stack([target + rng.normal(0, 1, 300), target + rng.normal(0, 1, 300)])
    blend = WeightedBlend().fit(predictions, target)
    assert (blend.weights_ >= -1e-9).all() and np.isclose(blend.weights_.sum(), 1)
    errors = np.abs(predictions - target[:, None]).mean(axis=0)
    assert np.abs(blend.predict(predictions) - target).mean() < errors.min()


def test_equal_blend_is_mean():
    predictions = np.array([[1.0, 3.0], [2.0, 6.0]])
    assert np.allclose(WeightedBlend(equal=True).fit(predictions, [0, 0]).predict(predictions), [2.0, 4.0])


def write_pair(directory, name, train_names, test_names, columns):
    """Сохраняет пару файлов прогнозов, как их пишет пайплайн."""
    oof = pd.DataFrame({"img_name": train_names, **{model: values for model, (values, _) in columns.items()}})
    test = pd.DataFrame({"img_name": test_names, **{model: values for model, (_, values) in columns.items()}})
    oof.to_csv(directory / f"oof_{name}.csv", index=False)
    test.to_csv(directory / f"test_{name}.csv", index=False)
    return {"oof": directory / f"oof_{name}.csv", "test": directory / f"test_{name}.csv"}


def test_read_predictions_rejects_repeated_model(tmp_path):
    first = write_pair(tmp_path, "a", ["x_1.png"], ["y_1.png"], {"m": ([1.0], [2.0])})
    second = write_pair(tmp_path, "b", ["x_1.png"], ["y_1.png"], {"m": ([1.0], [2.0])})
    with pytest.raises(ValueError):
        read_predictions([first, second], "img_name")


def test_read_predictions_rejects_missing_images(tmp_path):
    first = write_pair(tmp_path, "a", ["x_1.png", "x_2.png"], ["y_1.png"], {"m": ([1.0, 2.0], [2.0])})
    second = write_pair(tmp_path, "b", ["x_1.png"], ["y_1.png"], {"n": ([1.0], [2.0])})
    with pytest.raises(ValueError):
        read_predictions([first, second], "img_name")


def test_stacking_on_predictions_does_not_receive_features():
    columns = ["blur", PREDICTION_PREFIX + "cnn"]
    assert model_columns({"type": "blend", "inputs": "predictions"}, columns) == [PREDICTION_PREFIX + "cnn"]
    assert model_columns({"type": "lightgbm"}, columns) == columns


def test_run_ensemble_prefers_accurate_model(config, tmp_path):
    train = pd.read_csv(config["data"]["dir"] / "train.csv")
    test = pd.read_csv(config["data"]["dir"] / "sample_submission.csv")
    exact = (train["PSNR"].to_numpy(), np.full(len(test), 30.0))
    shifted = (train["PSNR"].to_numpy() + 2, np.full(len(test), 32.0))
    config["predictions"] = [write_pair(tmp_path, "base", train["img_name"], test["img_name"],
                                        {"exact": exact, "shifted": shifted})]
    config["models"] = {"mean": {"type": "blend", "inputs": "predictions", "params": {"equal": True}},
                        "weights": {"type": "blend", "inputs": "predictions"},
                        "stacking": {"type": "lightgbm", "params": {"n_estimators": 5, "min_child_samples": 2,
                                                                    "verbose": -1}}}

    results = run(config).set_index("model")

    assert results.loc["weights", "mae"] < 1e-6
    assert np.isclose(results.loc["mean", "mae"], 1.0)
    submission = pd.read_csv(config["output"]["dir"] / "submission.csv")
    assert np.allclose(submission["PSNR"], 30.0)


def test_run_rejects_predictions_for_other_images(config, tmp_path):
    test = pd.read_csv(config["data"]["dir"] / "sample_submission.csv")
    config["predictions"] = [write_pair(tmp_path, "base", ["0_1.png"], test["img_name"],
                                        {"m": ([28.0], np.full(len(test), 30.0))})]
    with pytest.raises(ValueError):
        run(config)
