"""Свёрточная нейросеть, которая предсказывает PSNR по самому изображению, с интерфейсом модели scikit-learn."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from src.features import find_images
from src.neural import learning_rate_schedule, make_optimizer, train_network

PREDICT_BATCH_SIZE = 8


def read_image(path):
    """Читает PNG в тензор float32 со значениями 0–255. AutoGraph не нужен: в функции нет ветвлений."""
    import tensorflow as tf

    return tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)


def output_layer():
    """Выход регрессии всегда в float32: при смешанной точности float16 огрубил бы прогноз."""
    import keras

    return keras.layers.Dense(1, dtype="float32")


def build_small_cnn(filters):
    """Свёрточная сеть с нуля: в каждом блоке две свёртки 3×3 с батч-нормализацией и ReLU,
    затем уменьшение размера вдвое; после блоков — усреднение по всему изображению и выход."""
    import keras

    layers = [keras.Input((None, None, 3)), keras.layers.Rescaling(1 / 255)]
    for count in filters:
        for _ in range(2):
            layers += [keras.layers.Conv2D(count, 3, padding="same", use_bias=False),
                       keras.layers.BatchNormalization(), keras.layers.Activation("relu")]
        layers.append(keras.layers.MaxPooling2D())
    layers += [keras.layers.GlobalAveragePooling2D(), output_layer()]
    return keras.Sequential(layers)


def build_efficientnet(pretrained):
    """EfficientNetB0 без классификатора, с весами ImageNet или случайными, и выходом для регрессии."""
    import keras

    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet" if pretrained else None,
                                                 input_shape=(None, None, 3), pooling="avg")
    return keras.Sequential([backbone, output_layer()])


class ImageCNNRegressor(BaseEstimator, RegressorMixin):
    """Свёрточная сеть на изображениях в исходном разрешении.

    На вход подаются имена файлов изображений, сами изображения читаются из images_dir. Изображения
    не уменьшаются: уменьшение стёрло бы размытие, шум и следы сжатия, по которым
    оценивается PSNR. Сеть заканчивается усреднением по всему изображению, поэтому принимает любой
    размер. backbone="small" — сеть с нуля из блоков с числом фильтров filters, backbone="efficientnetb0" —
    EfficientNetB0, при pretrained=True с весами, обученными на ImageNet. При flips=True обучающие
    изображения случайно отражаются по горизонтали и вертикали: это не меняет PSNR. Если задан
    crop=(высота, ширина), сеть обучается на случайных вырезках этого размера, а прогноз делает по целому
    изображению. При mixed_precision=True вычисления идут во float16 с масштабированием потерь, а веса
    хранятся во float32. Целевая переменная стандартизуется, как в MLPRegressor.
    """

    def __init__(self, images_dir="data", backbone="small", filters=(32, 64, 128, 256), pretrained=True,
                 optimizer="adamw", learning_rate=3e-4, scheduler="cosine", batch_size=16, epochs=10, loss="mae",
                 flips=True, crop=None, mixed_precision=False, random_state=0):
        self.images_dir = images_dir
        self.backbone = backbone
        self.filters = filters
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.flips = flips
        self.crop = crop
        self.mixed_precision = mixed_precision
        self.random_state = random_state

    def _paths(self, X):
        """Пути к изображениям в порядке имён из X."""
        return find_images(np.asarray(X).ravel().tolist(), self.images_dir)

    def _build(self):
        """Собирает сеть; политика точности задаётся только на время сборки, чтобы не влиять на другие модели."""
        import keras

        builders = {"small": lambda: build_small_cnn(self.filters),
                    "efficientnetb0": lambda: build_efficientnet(self.pretrained)}
        if self.backbone not in builders:
            raise ValueError(f"Неизвестная архитектура {self.backbone}, доступны: {sorted(builders)}")
        previous = keras.config.dtype_policy()
        keras.config.set_dtype_policy("mixed_float16" if self.mixed_precision else "float32")
        try:
            return builders[self.backbone]()
        finally:
            keras.config.set_dtype_policy(previous)

    def fit(self, X, y):
        """Обучает сеть на мини-батчах изображений, перемешивая их в каждой эпохе."""
        import keras
        import tensorflow as tf

        keras.utils.set_random_seed(self.random_state)
        self.model_ = self._build()
        target = np.asarray(y, dtype="float32")
        self.target_mean_, self.target_std_ = float(target.mean()), float(target.std())
        scaled = ((target - self.target_mean_) / self.target_std_).reshape(-1, 1)

        def load(path, value):
            image = read_image(path)
            if self.crop is not None:
                image = tf.image.random_crop(image, (*self.crop, 3))
            if self.flips:
                image = tf.image.random_flip_up_down(tf.image.random_flip_left_right(image))
            return image, value

        # Перемешиваются пути, а изображения читаются уже после: буфер перемешивания из раскодированных
        # изображений занял бы десятки гигабайт памяти.
        batches = (tf.data.Dataset.from_tensor_slices((self._paths(X), scaled))
                   .shuffle(len(target), seed=self.random_state)
                   .map(tf.autograph.experimental.do_not_convert(load), num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(self.batch_size).prefetch(tf.data.AUTOTUNE))

        steps = int(np.ceil(len(target) / self.batch_size)) * self.epochs
        optimizer = make_optimizer(self.optimizer, learning_rate_schedule(self.learning_rate, self.scheduler, steps))
        if self.mixed_precision:
            # Без масштабирования малые градиенты во float16 обращаются в ноль.
            optimizer = keras.optimizers.LossScaleOptimizer(optimizer)
        train_network(self.model_, batches, self.loss, optimizer, self.epochs)
        return self

    def predict(self, X):
        import tensorflow as tf

        batches = (tf.data.Dataset.from_tensor_slices(self._paths(X))
                   .map(tf.autograph.experimental.do_not_convert(read_image), num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(PREDICT_BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
        scaled = np.concatenate([self.model_(batch, training=False).numpy()[:, 0] for batch in batches])
        return scaled * self.target_std_ + self.target_mean_
