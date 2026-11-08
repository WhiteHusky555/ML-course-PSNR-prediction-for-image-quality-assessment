"""Создание моделей по описанию из конфигурации."""
from itertools import product

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_TYPES = {
    "dummy": DummyRegressor,
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}
SCALERS = {"standard": StandardScaler}


def build_model(spec):
    """Создаёт модель по типу и параметрам из раздела models.

    Если задан scaler, масштабирование и модель объединяются в конвейер: параметры масштаба
    оцениваются только на тех данных, на которых обучается модель, то есть на обучающей части фолда.
    """
    model_type = spec["type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Неизвестный тип модели {model_type}, доступны: {sorted(MODEL_TYPES)}")
    model = MODEL_TYPES[model_type](**spec.get("params", {}))
    scaler = spec.get("scaler")
    if scaler is None:
        return model
    if scaler not in SCALERS:
        raise ValueError(f"Неизвестный способ масштабирования {scaler}, доступны: {sorted(SCALERS)}")
    return make_pipeline(SCALERS[scaler](), model)


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
