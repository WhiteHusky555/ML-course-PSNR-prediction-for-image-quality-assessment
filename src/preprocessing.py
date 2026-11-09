"""Шаги предобработки признаков, которые обучаются на обучающей части фолда."""
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Ограничивает каждый признак квантилями, посчитанными на обучающих данных."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, features, target=None):
        self.bounds_ = np.quantile(np.asarray(features, dtype=float), [self.lower, self.upper], axis=0)
        return self

    def transform(self, features):
        return np.clip(np.asarray(features, dtype=float), *self.bounds_)


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Оставляет признаки, у которых модуль корреляции Спирмена с ранее оставленными не выше порога.

    Признаки просматриваются по порядку, поэтому из группы связанных остаётся первый.
    Постоянный признак даёт неопределённую корреляцию и отбрасывается.
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, features, target=None):
        ranks = rankdata(np.asarray(features, dtype=float), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.abs(np.corrcoef(ranks, rowvar=False))
        keep = []
        for column in range(correlation.shape[0]):
            if not np.isfinite(correlation[column, column]):
                continue
            if all(correlation[column, kept] <= self.threshold for kept in keep):
                keep.append(column)
        self.keep_ = np.array(keep)
        return self

    def transform(self, features):
        return np.asarray(features, dtype=float)[:, self.keep_]
