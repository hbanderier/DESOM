import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.metrics import Mean


@tf.keras.saving.register_keras_serializable(package="Desom", name="Kerasom")
class Kerasom(Model):
    def __init__(self, som_layer, sigma=2, **kwargs):
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
    
    