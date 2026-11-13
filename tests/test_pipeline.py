"""Проверки загрузки данных, разбиения на фолды и пайплайна на синтетических данных."""
import numpy as np
import pandas as pd
import pytest

from conftest import N_VARIANTS, make_labels
from main import run
from src.data import add_name_parts, attach_features, check_feature_table, feature_columns, load_data
from src.models import build_model
from src.validation import cross_validate, make_folds


def test_add_name_parts_splits_source_and_variant():
    frame = pd.DataFrame({"img_name": ["100090_1.png", "a_b_5.png"]})
    parts = add_name_parts(frame, "img_name")
    assert parts["source"].tolist() == ["100090", "a_b"]
    assert parts["variant"].tolist() == ["1", "5"]


def test_add_name_parts_rejects_unexpected_name():
    with pytest.raises(ValueError, match="шаблону"):
        add_name_parts(pd.DataFrame({"img_name": ["100090.png"]}), "img_name")


def test_load_data_rejects_missing_target(config):
    train_path = config["data"]["dir"] / "train.csv"
    labels = pd.read_csv(train_path)
    labels.loc[0, "PSNR"] = None
    labels.to_csv(train_path, index=False)
    with pytest.raises(ValueError, match="пропуски"):
        load_data(config["data"])


def test_load_data_rejects_duplicate_names(config):
    train_path = config["data"]["dir"] / "train.csv"
    labels = pd.read_csv(train_path)
    pd.concat([labels, labels.head(1)]).to_csv(train_path, index=False)
    with pytest.raises(ValueError, match="повторяющиеся"):
        load_data(config["data"])


def test_load_data_rejects_shared_sources(config):
    make_labels([0], 0).to_csv(config["data"]["dir"] / "sample_submission.csv", index=False)
    with pytest.raises(ValueError, match="Исходники"):
        load_data(config["data"])


def test_feature_columns_exclude_name_and_service_columns(config):
    train, _ = load_data(config["data"])
    assert feature_columns(train.assign(blur=0.0), "img_name", "PSNR") == ["blur"]


def test_check_feature_table_rejects_missing_image():
    table = pd.DataFrame({"img_name": ["a_1.png"], "blur": [1.0]})
    with pytest.raises(ValueError, match="нет 1 изображений"):
        check_feature_table(table, ["a_1.png", "a_2.png"], "img_name")


def test_check_feature_table_rejects_missing_value():
    table = pd.DataFrame({"img_name": ["a_1.png"], "blur": [np.nan]})
    with pytest.raises(ValueError, match="пропуски"):
        check_feature_table(table, ["a_1.png"], "img_name")


def test_attach_features_keeps_row_order():
    frame = pd.DataFrame({"img_name": ["b_1.png", "a_1.png"], "PSNR": [1.0, 2.0]})
    table = pd.DataFrame({"img_name": ["a_1.png", "b_1.png"], "blur": [20.0, 10.0]})
    attached = attach_features(frame, table, "img_name")
    assert attached["img_name"].tolist() == ["b_1.png", "a_1.png"]
    assert attached["blur"].tolist() == [10.0, 20.0]


def test_folds_keep_each_source_in_one_part_and_cover_all_rows(config):
    train, _ = load_data(config["data"])
    folds = make_folds(train, "PSNR", 3, 5, 42)
    for train_idx, valid_idx in folds:
        assert not set(train["source"].iloc[train_idx]) & set(train["source"].iloc[valid_idx])
    assert sorted(np.concatenate([valid_idx for _, valid_idx in folds])) == list(range(len(train)))


def test_folds_without_grouping_split_sources(config):
    train, _ = load_data(config["data"])
    folds = make_folds(train, "PSNR", 3, 5, 42, group=False)
    shared = [set(train["source"].iloc[t]) & set(train["source"].iloc[v]) for t, v in folds]
    assert any(shared)
    assert sorted(np.concatenate([valid_idx for _, valid_idx in folds])) == list(range(len(train)))


def test_run_saves_results_and_submission_of_best_model(config):
    results = run(config)
    train = pd.read_csv(config["data"]["dir"] / "train.csv")
    output_dir = config["output"]["dir"]
    saved = pd.read_csv(output_dir / "submission.csv")
    expected_names = pd.read_csv(config["data"]["dir"] / "sample_submission.csv")["img_name"]

    assert results["model"].tolist() == ["mean", "constant"]
    assert results.loc[1, "mae"] == pytest.approx(train["PSNR"].mean())
    assert {f"mae_{variant}" for variant in range(1, N_VARIANTS + 1)} <= set(results.columns)
    assert saved["img_name"].tolist() == expected_names.tolist()
    assert saved["PSNR"].to_numpy() == pytest.approx(train["PSNR"].mean())
    assert (output_dir / "results.csv").exists()


def test_run_evaluates_each_grid_point_and_submits_best(config):
    table = pd.read_csv(config["features"]["table"])
    table["blur"] = np.arange(len(table), dtype=float)
    table.to_csv(config["features"]["table"], index=False)
    config["models"] = {
        "median": {"type": "dummy", "params": {"strategy": "median"}},
        "ridge": {"type": "ridge", "scaler": "standard", "grid": {"alpha": [0.01, 1000.0]}},
    }

    results = run(config)
    saved = pd.read_csv(config["output"]["dir"] / "submission.csv")

    assert results["model"].tolist() == ["ridge alpha=0.01", "ridge alpha=1000.0", "median"]
    assert results.loc[0, "mae"] < 0.01
    assert saved["PSNR"].nunique() == len(saved)


def test_run_saves_predictions_of_every_model_for_ensembles(config):
    table = pd.read_csv(config["features"]["table"])
    table["blur"] = np.arange(len(table), dtype=float)
    table.to_csv(config["features"]["table"], index=False)
    config["models"] = {"median": {"type": "dummy", "params": {"strategy": "median"}},
                        "ridge": {"type": "ridge", "params": {"alpha": 1.0}}}

    results = run(config).set_index("model")
    output_dir = config["output"]["dir"]
    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    test = pd.read_csv(output_dir / "test_predictions.csv")
    submission = pd.read_csv(output_dir / "submission.csv")

    assert list(oof.columns) == ["img_name", "PSNR", "median", "ridge"]
    assert list(test.columns) == ["img_name", "median", "ridge"]
    for name in ("median", "ridge"):
        assert (oof["PSNR"] - oof[name]).abs().mean() == pytest.approx(results.loc[name, "mae"])
    assert submission["PSNR"].to_numpy() == pytest.approx(test[results.index[0]].to_numpy())


def test_results_show_overfitting_of_nearest_neighbour(config):
    rng = np.random.default_rng(0)
    table = pd.read_csv(config["features"]["table"])
    table["blur"] = rng.normal(size=len(table))
    table.to_csv(config["features"]["table"], index=False)
    config["models"] = {"knn": {"type": "knn", "params": {"n_neighbors": 1}}}

    results = run(config)

    assert results.loc[0, "train_mae"] == 0
    assert results.loc[0, "mae"] > 1


def test_parallel_folds_give_same_scores(config):
    train, _ = load_data(config["data"])
    train = train.assign(blur=np.random.default_rng(0).normal(size=len(train)))
    folds = make_folds(train, "PSNR", 3, 5, 42)
    spec = {"type": "gradboost", "params": {"n_estimators": 20, "random_state": 0}}
    sequential = cross_validate(lambda: build_model(spec), train, ["blur"], folds, "PSNR")
    parallel = cross_validate(lambda: build_model(spec), train, ["blur"], folds, "PSNR", n_jobs=2)
    assert np.array_equal(parallel.pop("predictions"), sequential.pop("predictions"))
    assert parallel == sequential


def test_run_with_tuning_compares_models_on_check_seeds(config):
    config["models"] = {"ridge": {"type": "ridge", "params": {"alpha": 1.0},
                                  "search": {"alpha": {"low": 0.1, "high": 10.0, "log": True}}}}
    config["tuning"] = {"n_trials": 3, "check_seeds": [1, 2]}

    results = run(config)

    assert sorted(results["model"]) == ["ridge", "ridge tuned"]
    assert "seed_std" in results
    assert (config["output"]["dir"] / "trials_ridge.csv").exists()
    assert (config["output"]["dir"] / "best_params.yaml").exists()
