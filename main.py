"""Запуск пайплайна: загрузка данных, кросс-валидация моделей, сабмит лучшей из них."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import attach_features, check_feature_table, feature_columns, load_data
from src.features import extract_features
from src.models import build_model, expand_grid, model_columns
from src.validation import cross_validate, make_folds

FEATURE_FLOAT_FORMAT = "%.7g"


def prepare_feature_table(features_config, names, id_column):
    """Читает готовую таблицу признаков или пересчитывает её по изображениям."""
    path = Path(features_config["table"])
    if features_config["recompute"]:
        table = extract_features(names, features_config["images_dir"], id_column)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format=FEATURE_FLOAT_FORMAT)
    else:
        table = pd.read_csv(path)
    check_feature_table(table, names, id_column)
    return table


def evaluate_models(models, config, train, columns, target):
    """Оценивает модели на общих фолдах и возвращает таблицу по возрастанию MAE."""
    validation = config["validation"]
    folds = make_folds(train, target, validation["n_splits"], validation["n_bins"], validation["seed"],
                       validation.get("group", True))
    rows = []
    for name, spec in models.items():
        scores = cross_validate(lambda: build_model(spec), train, model_columns(spec, columns), folds, target)
        print(f"{name}: MAE {scores['mae']:.4f}", flush=True)
        by_variant = {f"mae_{variant}": mae for variant, mae in scores["mae_by_variant"].items()}
        rows.append({"model": name, "mae": scores["mae"], "fold_std": scores["fold_std"],
                     "train_mae": scores["train_mae"], **by_variant})
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True)


def run(config):
    """Выполняет пайплайн и сохраняет таблицу результатов и сабмит лучшей модели."""
    id_column = config["data"]["id_column"]
    target = config["data"]["target"]
    train, test = load_data(config["data"])
    names = pd.concat([train[id_column], test[id_column]], ignore_index=True)
    table = prepare_feature_table(config["features"], names, id_column)
    train = attach_features(train, table, id_column)
    test = attach_features(test, table, id_column)
    columns = feature_columns(train, id_column, target)

    models = expand_grid(config["models"])
    results = evaluate_models(models, config, train, columns, target)
    best_spec = models[results.loc[0, "model"]]
    best_columns = model_columns(best_spec, columns)
    best_model = build_model(best_spec).fit(train[best_columns], train[target])
    submission = test[[id_column]].assign(**{target: best_model.predict(test[best_columns])})

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
