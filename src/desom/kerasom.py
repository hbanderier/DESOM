import tensorflow as tf
from keras.models import Model
from keras.metrics import Mean


@tf.keras.saving.register_keras_serializable(package="Desom", name="Desom")
class Kerasom(Model):
    def __init__(self, som_layer, sigma:float = 2., **kwargs):
        super().__init__(**kwargs)
        self.som_layer = som_layer
        self.loss_tracker = Mean(name="loss")
        self.sigma = sigma
        self.som_layer.sigma = self.sigma

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self(x, training=True)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    def test_step(self, x):
        loss = self(x, training=False)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    def call(self, inputs, **kwargs):
        return self.som_layer(inputs, self.sigma, **kwargs)


class SOMCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, speed=1, start_sigma=2):
        super().__init__()
        self.epoch_number = epochs
        self.speed = speed
        self.start_sigma = tf.constant(start_sigma, dtype=tf.float32)

    def on_train_begin(self, logs=None):
        if self.start_sigma is None:
            map_size = self.model.som_layer.map_size
            self.start_sigma = max(map_size[0], map_size[1]) / 2
        self.tau = self.epoch_number / self.speed
        self.model.sigma = tf.Variable(self.start_sigma, trainable=False, name='sigma', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.sigma, self.start_sigma * tf.math.exp(-epoch / self.tau))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['sigma'] = tf.keras.backend.get_value(self.model.sigma)
    