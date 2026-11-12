"""Полносвязная нейросеть на Keras с интерфейсом модели scikit-learn."""
import os

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Служебные сообщения TensorFlow о процессоре и оптимизациях не относятся к обучению модели.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SGD_MOMENTUM = 0.9


def make_optimizer(name, learning_rate):
    """Создаёт оптимизатор Keras по имени: adam, adamw, rmsprop или sgd (с моментом 0,9)."""
    import keras

    optimizers = {
        "adam": lambda: keras.optimizers.Adam(learning_rate),
        "adamw": lambda: keras.optimizers.AdamW(learning_rate),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate),
        "sgd": lambda: keras.optimizers.SGD(learning_rate, momentum=SGD_MOMENTUM),
    }
    if name not in optimizers:
        raise ValueError(f"Неизвестный оптимизатор {name}, доступны: {sorted(optimizers)}")
    return optimizers[name]()


def build_network(n_features, hidden_units, activation, batch_norm, dropout):
    """Собирает сеть: скрытые слои, после каждого — батч-нормализация, активация и dropout, затем выход."""
    import keras

    layers = [keras.Input((n_features,))]
    for units in hidden_units:
        layers.append(keras.layers.Dense(units, use_bias=not batch_norm))
        if batch_norm:
            layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Activation(activation))
        if dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)


class MLPRegressor(BaseEstimator, RegressorMixin):
    """Многослойный перцептрон для регрессии.

    hidden_units — размеры скрытых слоёв. Целевая переменная стандартизуется по обучающей выборке,
    а прогноз переводится обратно в дБ: так начальные веса подходят для любого масштаба цели.
    scheduler="cosine" уменьшает шаг обучения по косинусу от learning_rate до нуля за всё обучение,
    scheduler=None оставляет его постоянным. loss — функция потерь Keras: mae, mse или huber.
    """

    def __init__(self, hidden_units=(128,), activation="relu", batch_norm=False, dropout=0.0, optimizer="adam",
                 learning_rate=1e-3, scheduler=None, batch_size=256, epochs=100, loss="mae", random_state=0):
        self.hidden_units = hidden_units
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y):
        """Обучает сеть мини-батчами, перемешивая обучающую выборку в каждой эпохе.

        Цикл обучения написан явно вместо model.fit: в Keras 3.13 model.fit на каждом шаге переводит
        значения метрик в numpy способом, устаревшим для numpy 2, и выдаёт предупреждение.
        """
        import keras
        import tensorflow as tf

        if self.scheduler not in (None, "cosine"):
            raise ValueError(f"Неизвестный scheduler {self.scheduler}, доступны: None, cosine")
        loss_function = keras.losses.get(self.loss)
        keras.utils.set_random_seed(self.random_state)
        features = np.asarray(X, dtype="float32")
        target = np.asarray(y, dtype="float32")
        self.target_mean_, self.target_std_ = float(target.mean()), float(target.std())
        scaled = ((target - self.target_mean_) / self.target_std_).reshape(-1, 1)

        rate = self.learning_rate
        if self.scheduler == "cosine":
            steps = int(np.ceil(len(features) / self.batch_size)) * self.epochs
            rate = keras.optimizers.schedules.CosineDecay(self.learning_rate, steps)
        optimizer = make_optimizer(self.optimizer, rate)
        self.model_ = build_network(features.shape[1], self.hidden_units, self.activation, self.batch_norm,
                                    self.dropout)
        batches = (tf.data.Dataset.from_tensor_slices((features, scaled))
                   .shuffle(len(features), seed=self.random_state).batch(self.batch_size))

        # Без XLA-компиляции: она удерживает память процесса для каждой модели, а моделей обучаются сотни.
        # AutoGraph не нужен: в шаге нет ветвлений и циклов Python.
        @tf.function(jit_compile=False, autograph=False)
        def train_step(batch_features, batch_target):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(loss_function(batch_target, self.model_(batch_features, training=True)))
            gradients = tape.gradient(loss, self.model_.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model_.trainable_variables))

        for _ in range(self.epochs):
            for batch_features, batch_target in batches:
                train_step(batch_features, batch_target)
        return self

    def predict(self, X):
        scaled = self.model_(np.asarray(X, dtype="float32"), training=False).numpy()
        return scaled[:, 0] * self.target_std_ + self.target_mean_
