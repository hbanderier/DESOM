import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean


@tf.keras.saving.register_keras_serializable(package="Desom", name="Desom")
class Desom(Model):
    def __init__(self, encoder, decoder, som_layer, norm=1, factors=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.som_layer = som_layer
        if factors is None:
            factors = [1, 1, 1]
        self.factors = tf.constant(factors, dtype=tf.float32)
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.locality_error_tracker = Mean(name="locality_error")
        self.som_loss_tracker = Mean(name="som_loss")
        self.norm = norm
        self.norm_tensor = tf.constant(norm)
        self.sigma = 1.

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "som_layer": self.som_layer,
                "norm": self.norm,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["encoder"] = tf.keras.saving.deserialize_keras_object(config["encoder"])
        config["decoder"] = tf.keras.saving.deserialize_keras_object(config["decoder"])
        config["som_layer"] = tf.keras.saving.deserialize_keras_object(
            config["som_layer"]
        )
        return cls(**config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.locality_error_tracker,
            self.som_loss_tracker,
        ]

    @tf.function
    def compute_losses(self, x, y, xp, som_loss):
        reconstruction_loss = tf.math.reduce_mean(
            tf.math.square(x - xp) * self.norm_tensor, axis=(1, 2, 3)
        )
        D_data = tf.norm(
            tf.expand_dims(x[..., 0], axis=1) - tf.expand_dims(x[..., 0], axis=0), axis=(2, 3)
        )
        D_data = tf.math.divide_no_nan(D_data, tf.reduce_max(D_data))
        D_embeds = tf.norm(
            tf.expand_dims(y, axis=1) - tf.expand_dims(y, axis=0), axis=-1
        )
        D_embeds = tf.math.divide_no_nan(D_embeds, tf.reduce_max(D_embeds))
        locality_error = tf.reduce_sum(tf.square(D_embeds - D_data), axis=1)
        total_loss = (
            tf.math.multiply_no_nan(self.factors[0], reconstruction_loss) + 
            tf.math.multiply_no_nan(self.factors[1], locality_error) + 
            tf.math.multiply_no_nan(self.factors[2], som_loss)
        )
        return reconstruction_loss, locality_error, total_loss
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            y, xp, som_loss = self(x, training=True)
            reconstruction_loss, locality_error, total_loss = self.compute_losses(x, y, xp, som_loss)
                        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.locality_error_tracker.update_state(locality_error)
        self.som_loss_tracker.update_state(som_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction loss": self.reconstruction_loss_tracker.result(),
            "locality error": self.locality_error_tracker.result(),
            "som loss": self.som_loss_tracker.result(),
        }

    def test_step(self, x):
        y, xp, som_loss = self(x, training=False)
        reconstruction_loss, locality_error, total_loss = self.compute_losses(x, y, xp, som_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.locality_error_tracker.update_state(locality_error)
        self.som_loss_tracker.update_state(som_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction loss": self.reconstruction_loss_tracker.result(),
            "locality error": self.locality_error_tracker.result(),
            "som loss": self.som_loss_tracker.result(),
        }

    def call(self, x, **kwargs):
        y = self.encoder(x, **kwargs)
        som_loss = self.som_layer(y, self.sigma, **kwargs)
        xp = self.decoder(y, **kwargs)
        return y, xp, som_loss