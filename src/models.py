"""Создание моделей по описанию из конфигурации."""
from sklearn.dummy import DummyRegressor

MODEL_TYPES = {"dummy": DummyRegressor}


def build_model(spec):
    """Создаёт модель по типу и параметрам из раздела models."""
    model_type = spec["type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Неизвестный тип модели {model_type}, доступны: {sorted(MODEL_TYPES)}")
    return MODEL_TYPES[model_type](**spec.get("params", {}))
