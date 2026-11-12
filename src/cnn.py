"""Свёрточная нейросеть, которая предсказывает PSNR по самому изображению, с интерфейсом модели scikit-learn."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from src.features import find_images
from src.neural import learning_rate_schedule, make_optimizer, train_network

PREDICT_BATCH_SIZE = 8


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
    layers += [keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(1)]
    return keras.Sequential(layers)


def build_efficientnet(pretrained):
    """EfficientNetB0 без классификатора, с весами ImageNet или случайными, и выходом для регрессии."""
    import keras

    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet" if pretrained else None,
                                                 input_shape=(None, None, 3), pooling="avg")
    return keras.Sequential([backbone, keras.layers.Dense(1)])


class ImageCNNRegressor(BaseEstimator, RegressorMixin):
    """Свёрточная сеть на изображениях в исходном разрешении.

    На вход подаются имена файлов изображений, сами изображения читаются из images_dir. Изображения
    не уменьшаются и не обрезаются: уменьшение стёрло бы размытие, шум и следы сжатия, по которым
    оценивается PSNR. Сеть заканчивается усреднением по всему изображению, поэтому принимает любой
    размер. backbone="small" — сеть с нуля из блоков с числом фильтров filters, backbone="efficientnetb0" —
    EfficientNetB0, при pretrained=True с весами, обученными на ImageNet. При flips=True обучающие
    изображения случайно отражаются по горизонтали и вертикали: это не меняет PSNR. Целевая переменная
    стандартизуется, как в MLPRegressor.
    """

    def __init__(self, images_dir="data", backbone="small", filters=(32, 64, 128, 256), pretrained=True,
                 optimizer="adamw", learning_rate=3e-4, scheduler="cosine", batch_size=16, epochs=10, loss="mae",
                 flips=True, random_state=0):
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
        self.random_state = random_state

    def _images(self, X):
        """Набор изображений в порядке имён из X."""
        import tensorflow as tf

        paths = find_images(np.asarray(X).ravel().tolist(), self.images_dir)

        # Функции чтения и отражения не содержат ветвлений Python, поэтому AutoGraph им не нужен.
        @tf.autograph.experimental.do_not_convert
        def decode(path):
            return tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)

        return tf.data.Dataset.from_tensor_slices(paths).map(decode, num_parallel_calls=tf.data.AUTOTUNE)

    def _build(self):
        if self.backbone == "small":
            return build_small_cnn(self.filters)
        if self.backbone == "efficientnetb0":
            return build_efficientnet(self.pretrained)
        raise ValueError(f"Неизвестная архитектура {self.backbone}, доступны: small, efficientnetb0")

    def fit(self, X, y):
        """Обучает сеть на мини-батчах изображений, перемешивая их в каждой эпохе."""
        import keras
        import tensorflow as tf

        keras.utils.set_random_seed(self.random_state)
        self.model_ = self._build()
        target = np.asarray(y, dtype="float32")
        self.target_mean_, self.target_std_ = float(target.mean()), float(target.std())
        scaled = tf.data.Dataset.from_tensor_slices(((target - self.target_mean_) / self.target_std_).reshape(-1, 1))
        pairs = tf.data.Dataset.zip((self._images(X), scaled)).shuffle(len(target), seed=self.random_state)
        if self.flips:
            @tf.autograph.experimental.do_not_convert
            def flip(image, value):
                return tf.image.random_flip_up_down(tf.image.random_flip_left_right(image)), value

            pairs = pairs.map(flip, num_parallel_calls=tf.data.AUTOTUNE)
        batches = pairs.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        steps = int(np.ceil(len(target) / self.batch_size)) * self.epochs
        optimizer = make_optimizer(self.optimizer, learning_rate_schedule(self.learning_rate, self.scheduler, steps))
        train_network(self.model_, batches, self.loss, optimizer, self.epochs)
        return self

    def predict(self, X):
        import tensorflow as tf

        batches = self._images(X).batch(PREDICT_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        scaled = np.concatenate([self.model_(batch, training=False).numpy()[:, 0] for batch in batches])
        return scaled * self.target_std_ + self.target_mean_
