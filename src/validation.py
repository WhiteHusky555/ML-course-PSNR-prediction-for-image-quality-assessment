"""Разбиение данных на фолды и оценка моделей по кросс-валидации."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(train, target, n_splits, n_bins, seed):
    """Возвращает разбиения со стратификацией по бинам цели и группировкой по исходнику.

    Варианты одного исходника похожи между собой, а в test исходники из train не встречаются.
    Поэтому все варианты исходника должны попадать в один фолд.
    """
    bins = pd.qcut(train[target], n_bins, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(train, bins, train["source"]))


def cross_validate(build_model, train, columns, folds, target):
    """Считает MAE по out-of-fold предсказаниям, в целом и по типам искажений."""
    predictions = np.zeros(len(train))
    fold_scores = []
    for train_idx, valid_idx in folds:
        part = train.iloc[train_idx]
        valid = train.iloc[valid_idx]
        predictions[valid_idx] = build_model().fit(part[columns], part[target]).predict(valid[columns])
        fold_scores.append(mean_absolute_error(valid[target], predictions[valid_idx]))
    errors = (train[target] - predictions).abs()
    return {
        "mae": float(errors.mean()),
        "fold_std": float(np.std(fold_scores)),
        "mae_by_variant": errors.groupby(train["variant"]).mean().to_dict(),
    }
