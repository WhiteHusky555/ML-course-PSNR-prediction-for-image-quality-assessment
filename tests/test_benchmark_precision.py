"""Проверки сравнения float32 и смешанной точности."""
import numpy as np

from benchmark_precision import benchmark, prediction_gap


def test_benchmark_measures_both_precisions():
    table = benchmark(["small"], {"tiny": ((32, 32), 4)}, warmup_steps=1, measured_steps=3)
    assert table["precision"].tolist() == ["float32", "mixed_float16"]
    assert (table["images_per_second"] > 0).all()


def test_mixed_precision_prediction_close_to_float32():
    import tensorflow as tf

    images = tf.random.uniform((8, 32, 32, 3), 0, 255, seed=0)
    gap = prediction_gap("small", images)
    assert 0 < gap["mean_gap"] < 0.05 and np.isfinite(gap["max_gap"])
