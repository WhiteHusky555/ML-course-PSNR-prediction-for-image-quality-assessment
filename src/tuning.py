"""Подбор гиперпараметров моделей с помощью Optuna."""
import optuna
import yaml

from src.models import build_model, model_columns
from src.validation import cross_validate


def suggest_params(trial, space):
    """Выбирает значения параметров из пространства поиска, описанного в разделе search модели.

    Параметр со списком choices выбирается из списка. Параметр с границами low и high целый, если обе
    границы целые, иначе вещественный; log задаёт логарифмическую шкалу, step — шаг для целых значений.
    """
    params = {}
    for name, bounds in space.items():
        if "choices" in bounds:
            params[name] = trial.suggest_categorical(name, bounds["choices"])
        elif isinstance(bounds["low"], int) and isinstance(bounds["high"], int):
            params[name] = trial.suggest_int(name, bounds["low"], bounds["high"], step=bounds.get("step", 1),
                                             log=bounds.get("log", False))
        else:
            params[name] = trial.suggest_float(name, bounds["low"], bounds["high"], log=bounds.get("log", False))
    return params


def tune_model(spec, train, columns, folds, target, tuning, seed):
    """Ищет параметры модели, минимизирующие MAE на фолдах.

    Первым испытанием проверяются исходные параметры модели, поэтому найденный набор на этих фолдах
    не хуже исходного. Возвращает исследование Optuna со всеми испытаниями.
    """
    space = spec["search"]
    columns = model_columns(spec, columns)

    def objective(trial):
        point = {**spec, "params": {**spec.get("params", {}), **suggest_params(trial, space)}}
        return cross_validate(lambda: build_model(point), train, columns, folds, target,
                              spec.get("fold_jobs", 1))["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    start = {name: value for name, value in spec.get("params", {}).items() if name in space}
    if start:
        study.enqueue_trial(start)
    study.optimize(objective, n_trials=tuning["n_trials"], timeout=tuning.get("timeout"))
    return study


def tune_models(models, train, columns, folds, target, tuning, seed, output_dir):
    """Подбирает параметры для всех моделей с разделом search.

    Возвращает описания подобранных моделей с суффиксом «tuned» в имени. История испытаний каждой
    модели сохраняется в trials_<модель>.csv, найденные параметры — в best_params.yaml.
    """
    tuned, best = {}, {}
    for name, spec in models.items():
        if "search" not in spec:
            continue
        study = tune_model(spec, train, columns, folds, target, tuning, seed)
        study.trials_dataframe().to_csv(output_dir / f"trials_{name}.csv", index=False)
        print(f"{name}: {len(study.trials)} испытаний, лучшая MAE на фолдах поиска {study.best_value:.4f}", flush=True)
        best[name] = study.best_params
        tuned[f"{name} tuned"] = {**spec, "params": {**spec.get("params", {}), **study.best_params}}
    (output_dir / "best_params.yaml").write_text(yaml.safe_dump(best, allow_unicode=True), encoding="utf-8")
    return tuned
