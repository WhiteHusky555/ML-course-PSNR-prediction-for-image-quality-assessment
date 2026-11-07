"""Запуск пайплайна: загрузка данных, кросс-валидация моделей, сабмит лучшей из них."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import feature_columns, load_data
from src.models import build_model
from src.validation import cross_validate, make_folds


def evaluate_models(config, train, columns, target):
    """Оценивает все модели из конфигурации и возвращает таблицу по возрастанию MAE."""
    validation = config["validation"]
    folds = make_folds(train, target, validation["n_splits"], validation["n_bins"], validation["seed"])
    rows = []
    for name, spec in config["models"].items():
        scores = cross_validate(lambda: build_model(spec), train, columns, folds, target)
        by_variant = {f"mae_{variant}": mae for variant, mae in scores["mae_by_variant"].items()}
        rows.append({"model": name, "mae": scores["mae"], "fold_std": scores["fold_std"], **by_variant})
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True)


def run(config):
    """Выполняет пайплайн и сохраняет таблицу результатов и сабмит лучшей модели."""
    id_column = config["data"]["id_column"]
    target = config["data"]["target"]
    train, test = load_data(config["data"])
    columns = feature_columns(train, id_column, target)

    results = evaluate_models(config, train, columns, target)
    best_name = results.loc[0, "model"]
    best_model = build_model(config["models"][best_name]).fit(train[columns], train[target])
    submission = test[[id_column]].assign(**{target: best_model.predict(test[columns])})

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)
    submission.to_csv(output_dir / "submission.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="путь к файлу конфигурации")
    args = parser.parse_args()
    print(run(load_config(args.config)).to_string(index=False))


if __name__ == "__main__":
    main()
