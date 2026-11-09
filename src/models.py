"""Создание моделей по описанию из конфигурации."""
from itertools import product

from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, QuantileTransformer, RobustScaler,
                                   StandardScaler, TargetEncoder)

from src.preprocessing import CorrelationFilter, QuantileClipper

MODEL_TYPES = {
    "dummy": DummyRegressor,
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}
SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "quantile": lambda: QuantileTransformer(output_distribution="normal", random_state=0),
}
ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": OrdinalEncoder,
    "target": lambda: TargetEncoder(target_type="continuous", random_state=0),
}
VARIANT_COLUMN = "variant"


def choose(table, name, kind):
    """Возвращает элемент словаря по имени или сообщает, какие имена допустимы."""
    if name not in table:
        raise ValueError(f"Неизвестный {kind} {name}, доступны: {sorted(table)}")
    return table[name]


def preprocessing_steps(spec):
    """Шаги предобработки числовых признаков в порядке применения.

    clip — квантили для ограничения значений, select — порог корреляции для отбора признаков,
    scaler — способ масштабирования, pca — доля объяснённой дисперсии или число компонент.
    """
    steps = []
    if "clip" in spec:
        steps.append(QuantileClipper(*spec["clip"]))
    if "select" in spec:
        steps.append(CorrelationFilter(spec["select"]))
    if "scaler" in spec:
        steps.append(choose(SCALERS, spec["scaler"], "способ масштабирования")())
    if "pca" in spec:
        steps.append(PCA(spec["pca"], random_state=0))
    return steps


def build_model(spec):
    """Создаёт модель по типу и параметрам из раздела models.

    Предобработка и модель объединяются в конвейер, поэтому всё, что оценивается по данным
    (квантили, корреляции, масштаб, главные компоненты, кодировка), обучается только на обучающей
    части фолда. При заданном variant номер варианта кодируется и подаётся на вход модели; это
    используется только в исследовании кодирования категориальных признаков.
    """
    model = choose(MODEL_TYPES, spec["type"], "тип модели")(**spec.get("params", {}))
    steps = preprocessing_steps(spec)
    if "variant" in spec:
        encoder = choose(ENCODERS, spec["variant"], "способ кодирования")()
        numeric = make_pipeline(*steps) if steps else "passthrough"
        steps = [make_column_transformer((encoder, [VARIANT_COLUMN]), remainder=numeric)]
    return make_pipeline(*steps, model) if steps else model


def model_columns(spec, columns):
    """Столбцы, которые подаются модели: признаки и, если задано кодирование, номер варианта."""
    return [*columns, VARIANT_COLUMN] if "variant" in spec else list(columns)


def expand_grid(models_config):
    """Разворачивает сетки параметров: каждая комбинация значений из grid становится отдельной моделью.

    Значения из grid дополняют params и имеют приоритет над ними. Имя модели из сетки содержит
    значения её параметров, например «ridge alpha=10.0».
    """
    expanded = {}
    for name, spec in models_config.items():
        grid = spec.get("grid", {})
        base = {key: value for key, value in spec.items() if key != "grid"}
        for values in product(*grid.values()):
            point = dict(zip(grid, values))
            label = " ".join([name, *(f"{key}={value}" for key, value in point.items())])
            expanded[label] = {**base, "params": {**spec.get("params", {}), **point}}
    return expanded
