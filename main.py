"""Запуск пайплайна: загрузка данных, кросс-валидация моделей, сабмит лучшей из них."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import attach_features, check_feature_table, feature_columns, load_data
from src.features import extract_features
from src.models import build_model, expand_grid, model_columns, uses_images
from src.tuning import tune_models
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


def config_folds(config, train, target, seed):
    """Разбиение на фолды по разделу validation с заданным seed."""
    validation = config["validation"]
    return make_folds(train, target, validation["n_splits"], validation["n_bins"], seed, validation.get("group", True))


def evaluate_models(models, folds, train, test, columns, target):
    """Оценивает модели на общих фолдах.

    Возвращает таблицу по возрастанию MAE и для каждой модели пару из out-of-fold предсказаний
    на train и среднего прогноза моделей фолдов на test.
    """
    rows, predictions = [], {}
    for name, spec in models.items():
        scores = cross_validate(lambda: build_model(spec), train, model_columns(spec, columns), folds, target,
                                spec.get("fold_jobs", 1), test)
        print(f"{name}: MAE {scores['mae']:.4f}", flush=True)
        by_variant = {f"mae_{variant}": mae for variant, mae in scores["mae_by_variant"].items()}
        rows.append({"model": name, "mae": scores["mae"], "fold_std": scores["fold_std"],
                     "train_mae": scores["train_mae"], **by_variant})
        predictions[name] = scores["predictions"], scores["test_predictions"]
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True), predictions


def save_predictions(predictions, train, test, id_column, target, output_dir):
    """Сохраняет out-of-fold предсказания и прогнозы на test всех моделей: по столбцу на модель."""
    oof = train[[id_column, target]].assign(**{name: values for name, (values, _) in predictions.items()})
    test_table = test[[id_column]].assign(**{name: values for name, (_, values) in predictions.items()})
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_table.to_csv(output_dir / "test_predictions.csv", index=False)


def average_over_seeds(tables):
    """Усредняет таблицы результатов, полученные на разных разбиениях.

    При нескольких разбиениях добавляется столбец seed_std — разброс MAE между ними.
    """
    if len(tables) == 1:
        return tables[0]
    combined = pd.concat(tables).groupby("model", sort=False)
    results = combined.mean().assign(seed_std=combined["mae"].std())
    return results.sort_values("mae").reset_index()


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

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = config["features"]["images_dir"]
    models = {name: {**spec, "params": {**spec.get("params", {}), "images_dir": images_dir}} if uses_images(spec)
              else spec for name, spec in expand_grid(config["models"]).items()}
    seeds = [config["validation"]["seed"]]
    if "tuning" in config:
        # Параметры ищутся на фолдах основного seed, а модели сравниваются на фолдах других seed,
        # чтобы оценка подобранных моделей не была завышена отбором по тем же разбиениям.
        search_folds = config_folds(config, train, target, seeds[0])
        models.update(tune_models(models, train, columns, search_folds, target, config["tuning"], seeds[0],
                                  output_dir))
        seeds = config["tuning"]["check_seeds"]
    evaluated = [evaluate_models(models, config_folds(config, train, target, seed), train, test, columns, target)
                 for seed in seeds]
    results = average_over_seeds([table for table, _ in evaluated])
    # Прогнозы сохраняются по первому разбиению: на нём модели обучены, и out-of-fold предсказания согласованы.
    predictions = evaluated[0][1]
    save_predictions(predictions, train, test, id_column, target, output_dir)
    # Сабмит — среднее прогнозов моделей пяти фолдов лучшей модели: отдельное обучение на всём train не нужно.
    submission = test[[id_column]].assign(**{target: predictions[results.loc[0, "model"]][1]})

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
