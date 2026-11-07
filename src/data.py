"""Загрузка и проверка меток соревнования PSNR-IQA."""
from pathlib import Path

import numpy as np
import pandas as pd

NAME_PATTERN = r"^(?P<source>.+)_(?P<variant>\d+)\.png$"
SERVICE_COLUMNS = ("source", "variant")


def add_name_parts(frame, id_column):
    """Добавляет столбцы исходника и типа искажения, разобрав имя файла.

    Эти столбцы нужны для группировки фолдов и разбивки ошибки, на вход модели они не подаются.
    """
    parts = frame[id_column].str.extract(NAME_PATTERN)
    invalid = frame.loc[parts["source"].isna(), id_column]
    if not invalid.empty:
        raise ValueError(f"Имена файлов не соответствуют шаблону: {invalid.head().tolist()}")
    return frame.assign(source=parts["source"], variant=parts["variant"])


def check_columns(frame, columns, name):
    """Проверяет наличие столбцов и отсутствие повторов в первом из них."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"В файле {name} нет столбцов: {missing}")
    if frame[columns[0]].duplicated().any():
        raise ValueError(f"В файле {name} есть повторяющиеся значения {columns[0]}")


def check_target(train, target):
    """Проверяет, что целевая переменная числовая и конечная."""
    values = pd.to_numeric(train[target], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"Столбец {target} содержит пропуски или нечисловые значения")


def check_sources_disjoint(train, test):
    """Проверяет, что исходные изображения train и test не пересекаются."""
    shared = set(train["source"]) & set(test["source"])
    if shared:
        raise ValueError(f"Исходники встречаются и в train, и в test: {sorted(shared)[:5]}")


def load_data(data_config):
    """Загружает и проверяет train и test, добавляя служебные столбцы."""
    data_dir = Path(data_config["dir"])
    id_column = data_config["id_column"]
    target = data_config["target"]

    train = pd.read_csv(data_dir / data_config["train_file"])
    test = pd.read_csv(data_dir / data_config["test_file"])
    check_columns(train, [id_column, target], data_config["train_file"])
    check_columns(test, [id_column], data_config["test_file"])
    check_target(train, target)

    train = add_name_parts(train, id_column)
    test = add_name_parts(test, id_column)
    check_sources_disjoint(train, test)
    return train, test


def feature_columns(train, id_column, target):
    """Возвращает столбцы, которые подаются на вход модели."""
    excluded = {id_column, target, *SERVICE_COLUMNS}
    return [column for column in train.columns if column not in excluded]
