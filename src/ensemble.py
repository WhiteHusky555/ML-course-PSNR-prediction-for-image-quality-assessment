"""Ансамбль из прогнозов уже обученных моделей: взвешенное среднее и стекинг."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from sklearn.base import BaseEstimator, RegressorMixin

# LightGBM не принимает в именах признаков символы JSON, в том числе двоеточие.
PREDICTION_PREFIX = "pred_"


class WeightedBlend(BaseEstimator, RegressorMixin):
    """Взвешенное среднее прогнозов базовых моделей с неотрицательными весами, дающими в сумме 1.

    Веса минимизируют MAE на обучающих данных. Задача сводится к линейному программированию:
    ошибка каждой строки раскладывается на положительную и отрицательную части, и минимизируется
    их сумма. При equal=True веса не подбираются, все модели входят с равным весом.
    """

    def __init__(self, equal=False):
        self.equal = equal

    def fit(self, X, y):
        predictions = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        rows, models = predictions.shape
        if self.equal:
            self.weights_ = np.full(models, 1 / models)
            return self
        # Переменные: веса моделей, затем положительные и отрицательные части ошибок строк.
        cost = np.concatenate([np.zeros(models), np.ones(2 * rows)])
        errors = hstack([csr_matrix(predictions), -eye(rows), eye(rows)])
        weights_sum = csr_matrix(np.concatenate([np.ones(models), np.zeros(2 * rows)]))
        result = linprog(cost, A_eq=vstack([errors, weights_sum]), b_eq=np.append(target, 1.0), bounds=(0, None),
                         method="highs")
        if not result.success:
            raise RuntimeError(f"Веса ансамбля не найдены: {result.message}")
        self.weights_ = result.x[:models]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights_


def read_predictions(files, id_column):
    """Читает out-of-fold прогнозы на train и прогнозы на test из пар файлов (oof, test).

    Возвращает две таблицы: имя изображения и по столбцу на каждую модель. Столбцы моделей получают
    префикс, чтобы не совпасть с признаками. Имена моделей и изображений не должны повторяться,
    пропуски недопустимы.
    """
    oof_parts, test_parts = [], []
    for pair in files:
        oof = pd.read_csv(pair["oof"])
        test = pd.read_csv(pair["test"])
        models = [column for column in test.columns if column != id_column]
        if sorted(models) != sorted(column for column in oof.columns if column in models):
            raise ValueError(f"Модели в {pair['oof']} и {pair['test']} не совпадают")
        rename = {model: PREDICTION_PREFIX + model for model in models}
        oof_parts.append(oof[[id_column, *models]].rename(columns=rename).set_index(id_column))
        test_parts.append(test[[id_column, *models]].rename(columns=rename).set_index(id_column))
    oof, test = pd.concat(oof_parts, axis=1, join="inner"), pd.concat(test_parts, axis=1, join="inner")
    if oof.columns.duplicated().any():
        raise ValueError(f"Повторяющиеся модели: {oof.columns[oof.columns.duplicated()].tolist()}")
    if len(oof) != len(oof_parts[0]) or len(test) != len(test_parts[0]) or oof.isna().any().any():
        raise ValueError("Файлы прогнозов покрывают разные изображения или содержат пропуски")
    return oof.reset_index(), test.reset_index()
