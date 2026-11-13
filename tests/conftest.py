"""Общие фикстуры тестов: синтетические файлы данных и конфигурация пайплайна для них."""
import numpy as np
import pandas as pd
import pytest

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
    names = pd.concat([make_labels(range(30)), make_labels(range(100, 104), 0)])["img_name"]
    pd.DataFrame({"img_name": names, "blur": 1.0}).to_csv(tmp_path / "features.csv", index=False)
    return {
        "data": {
            "dir": tmp_path,
            "train_file": "train.csv",
            "test_file": "sample_submission.csv",
            "id_column": "img_name",
            "target": "PSNR",
        },
        "features": {"table": tmp_path / "features.csv", "recompute": False, "images_dir": tmp_path},
        "validation": {"n_splits": 3, "n_bins": 5, "seed": 42},
        "models": {
            "mean": {"type": "dummy", "params": {"strategy": "mean"}},
            "constant": {"type": "dummy", "params": {"strategy": "constant", "constant": 0.0}},
        },
        "output": {"dir": tmp_path / "outputs"},
    }
