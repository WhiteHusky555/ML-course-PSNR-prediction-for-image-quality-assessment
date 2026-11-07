"""Проверки загрузки данных, разбиения на фолды и пайплайна на синтетических данных."""
import numpy as np
import pandas as pd
import pytest

from main import run
from src.data import add_name_parts, feature_columns, load_data
from src.validation import make_folds

N_VARIANTS = 5


def make_labels(sources, target_value=None):
    """Строит таблицу меток: по пять вариантов на каждый исходник."""
    names = [f"{source}_{variant}.png" for source in sources for variant in range(1, N_VARIANTS + 1)]
    target = np.linspace(20, 40, len(names)) if target_value is None else target_value
    return pd.DataFrame({"img_name": names, "PSNR": target})


@pytest.fixture
def config(tmp_path):
    """Записывает корректные файлы данных и возвращает конфигурацию для них."""
    make_labels(range(30)).to_csv(tmp_path / "train.csv", index=False)
    make_labels(range(100, 104), 0).to_csv(tmp_path / "sample_submission.csv", index=False)
    return {
        "data": {
            "dir": tmp_path,
            "train_file": "train.csv",
            "test_file": "sample_submission.csv",
            "id_column": "img_name",
            "target": "PSNR",
        },
        "validation": {"n_splits": 3, "n_bins": 5, "seed": 42},
        "models": {
            "mean": {"type": "dummy", "params": {"strategy": "mean"}},
            "constant": {"type": "dummy", "params": {"strategy": "constant", "constant": 0.0}},
        },
        "output": {"dir": tmp_path / "outputs"},
    }


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


def test_folds_keep_each_source_in_one_part_and_cover_all_rows(config):
    train, _ = load_data(config["data"])
    folds = make_folds(train, "PSNR", 3, 5, 42)
    for train_idx, valid_idx in folds:
        assert not set(train["source"].iloc[train_idx]) & set(train["source"].iloc[valid_idx])
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
