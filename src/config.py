"""Чтение конфигурации эксперимента."""
from pathlib import Path

import yaml

REQUIRED_SECTIONS = ("data", "validation", "models", "output")


def load_config(path):
    """Читает YAML-файл и проверяет наличие обязательных разделов."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Файл конфигурации {path} не содержит словарь")
    missing = [section for section in REQUIRED_SECTIONS if not config.get(section)]
    if missing:
        raise ValueError(f"В файле конфигурации {path} нет разделов: {missing}")
    return config
