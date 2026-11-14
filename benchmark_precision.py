"""Сравнение обучения CNN во float32 и со смешанной точностью: скорость шага, пиковая память GPU
и расхождение прогнозов одной и той же сети в двух режимах."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn import ImageCNNRegressor, read_image
from src.neural import make_optimizer, train_network

# Размер входа и мини-батч те же, что в experiments/cnn.yaml.
SETUPS = {"full": ((480, 640), 16), "crop": ((256, 256), 32)}
BACKBONES = ["small", "efficientnetb0"]
WARMUP_STEPS, MEASURED_STEPS = 5, 30
GAP_IMAGES = 16


def build(backbone, mixed_precision):
    """Сеть в том виде, в каком её собирает ImageCNNRegressor для experiments/cnn.yaml."""
    return ImageCNNRegressor(backbone=backbone, mixed_precision=mixed_precision)._build()


def step_seconds(model, images, mixed_precision, warmup_steps, measured_steps):
    """Медианное время шага обучения тем же циклом, что в основном пайплайне, на одном батче в памяти:
    чтение PNG исключено, измеряются только вычисления. Первые warmup_steps шагов (трассировка графа,
    подбор алгоритмов свёртки) не учитываются."""
    import keras
    import tensorflow as tf

    optimizer = make_optimizer("adamw", 3e-4)
    if mixed_precision:
        optimizer = keras.optimizers.LossScaleOptimizer(optimizer)
    target = tf.zeros((len(images), 1))
    moments = []

    def batches():
        for _ in range(warmup_steps + measured_steps + 1):
            model.trainable_variables[0].numpy()  # GPU работает асинхронно: ждём окончания предыдущего шага
            moments.append(time.perf_counter())
            yield images, target

    train_network(model, batches(), "mae", optimizer, 1)
    return float(np.median(np.diff(moments[warmup_steps:])))


def gpu_available():
    import tensorflow as tf

    return bool(tf.config.list_physical_devices("GPU"))


def benchmark(backbones=BACKBONES, setups=SETUPS, warmup_steps=WARMUP_STEPS, measured_steps=MEASURED_STEPS):
    """Секунды на шаг, изображения в секунду и пиковая память GPU для каждой сети, входа и точности."""
    import keras
    import tensorflow as tf

    rows = []
    for backbone in backbones:
        for setup, (size, batch_size) in setups.items():
            images = tf.random.uniform((batch_size, *size, 3), 0, 255, seed=0)
            for mixed_precision in (False, True):
                keras.backend.clear_session()
                keras.utils.set_random_seed(0)
                model = build(backbone, mixed_precision)
                if gpu_available():
                    tf.config.experimental.reset_memory_stats("GPU:0")
                step = step_seconds(model, images, mixed_precision, warmup_steps, measured_steps)
                peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20 if gpu_available() else None
                rows.append({"backbone": backbone, "input": setup, "batch_size": batch_size,
                             "precision": "mixed_float16" if mixed_precision else "float32",
                             "step_seconds": step, "images_per_second": batch_size / step, "peak_memory_mb": peak})
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def prediction_gap(backbone, images):
    """Насколько прогноз сети со смешанной точностью отличается от прогноза той же сети во float32
    при одинаковых весах: среднее и максимальное отклонение в долях разброса прогнозов float32."""
    import keras

    keras.backend.clear_session()
    keras.utils.set_random_seed(0)
    reference = build(backbone, False)
    mixed = build(backbone, True)
    mixed.set_weights(reference.get_weights())
    exact = reference(images, training=False).numpy()[:, 0]
    approximate = mixed(images, training=False).numpy()[:, 0]
    gap = np.abs(exact - approximate) / exact.std()
    return {"backbone": backbone, "mean_gap": gap.mean(), "max_gap": gap.max()}


def load_images(images_dir, count=GAP_IMAGES):
    """Первые count изображений PNG из каталога в исходном разрешении."""
    import tensorflow as tf

    paths = sorted(Path(images_dir).rglob("*.png"))[:count]
    return tf.stack([read_image(str(path)) for path in paths])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, help="каталог с изображениями соревнования")
    parser.add_argument("--output-dir", default="outputs/precision")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speed = benchmark()
    reference = speed[speed["precision"] == "float32"].set_index(["backbone", "input"])["step_seconds"]
    speed["speedup"] = [reference[(row.backbone, row.input)] / row.step_seconds for row in speed.itertuples()]
    speed.to_csv(output_dir / "precision_speed.csv", index=False)

    images = load_images(args.images_dir)
    gaps = pd.DataFrame([prediction_gap(backbone, images) for backbone in BACKBONES])
    gaps.to_csv(output_dir / "precision_gap.csv", index=False)
    print(speed.to_string(index=False), gaps.to_string(index=False), sep="\n\n")


if __name__ == "__main__":
    main()
