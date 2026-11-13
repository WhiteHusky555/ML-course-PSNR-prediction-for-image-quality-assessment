"""Разбиение данных на фолды и оценка моделей по кросс-валидации."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_folds(train, target, n_splits, n_bins, seed, group=True):
    """Возвращает разбиения со стратификацией по бинам цели и группировкой по исходнику.

    Варианты одного исходника похожи между собой, а в test исходники из train не встречаются.
    Поэтому все варианты исходника должны попадать в один фолд. Разбиение без группировки
    (group=False) нужно только для сравнения: оно завышает качество моделей.
    """
    bins = pd.qcut(train[target], n_bins, labels=False, duplicates="drop")
    if not group:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(train, bins))
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(train, bins, train["source"]))


def fit_fold(build_model, train, columns, target, train_idx, valid_idx, test=None):
    """Обучает модель на обучающей части фолда; возвращает прогноз проверочной части, MAE на обучении
    и прогноз на test (None, если test не передан)."""
    part = train.iloc[train_idx]
    model = build_model().fit(part[columns], part[target])
    train_mae = mean_absolute_error(part[target], model.predict(part[columns]))
    test_predictions = None if test is None else model.predict(test[columns])
    return model.predict(train.iloc[valid_idx][columns]), train_mae, test_predictions


def cross_validate(build_model, train, columns, folds, target, n_jobs=1, test=None):
    """Считает MAE по out-of-fold предсказаниям, в целом и по типам искажений.

    train_mae — средняя по фолдам ошибка на обучающей части; разница с mae показывает переобучение.
    predictions — сами out-of-fold предсказания; если передан test, test_predictions — среднее прогнозов
    моделей всех фолдов на test. Те и другие нужны для ансамбля без повторного обучения моделей.
    При n_jobs > 1 фолды обучаются параллельно; это нужно однопоточным моделям, результат не меняется.
    """
    fitted = Parallel(n_jobs)(delayed(fit_fold)(build_model, train, columns, target, train_idx, valid_idx, test)
                              for train_idx, valid_idx in folds)
    predictions = np.zeros(len(train))
    fold_scores = []
    for (_, valid_idx), (fold_predictions, _, _) in zip(folds, fitted):
        predictions[valid_idx] = fold_predictions
        fold_scores.append(mean_absolute_error(train[target].iloc[valid_idx], fold_predictions))
    errors = (train[target] - predictions).abs()
    return {
        "mae": float(errors.mean()),
        "fold_std": float(np.std(fold_scores)),
        "train_mae": float(np.mean([train_mae for _, train_mae, _ in fitted])),
        "mae_by_variant": errors.groupby(train["variant"]).mean().to_dict(),
        "predictions": predictions,
        "test_predictions": None if test is None else np.mean([fold_test for _, _, fold_test in fitted], axis=0),
    }
