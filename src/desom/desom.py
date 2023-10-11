import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy


class DESOM(Model):
    def __init__(self, encoder, decoder, som_layer, factors=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.som_layer = som_layer
        if factors is None:
            factors = [1, 0.1, 1]
        self.factors = factors
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.orthogonal_loss_tracker = Mean(name="orthogonal_loss")
        self.som_loss_tracker = Mean(name="som_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.orthogonal_loss_tracker,
            self.som_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            som_loss = self.som_layer(z)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            embeddings = self.decoder(tf.eye(self.decoder.input.shape[-1]))
            embeddings = tf.reshape(embeddings, (embeddings.shape[0], -1))
            orthogonal_loss = tf.reduce_sum(tf.square(embeddings @ tf.transpose(embeddings) - tf.eye(embeddings.shape[0])))
            total_loss = self.factors[0] * reconstruction_loss + self.factors[1] * orthogonal_loss + self.factors[2] * som_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.orthogonal_loss_tracker.update_state(orthogonal_loss)
        self.som_loss_tracker.update_state(som_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "orthogonal_loss": self.orthogonal_loss_tracker.result(),
            "som_loss": self.som_loss_tracker.result(),
        }