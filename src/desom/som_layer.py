"""
Implementation of the Deep Embedded Self-Organizing Map model
SOM layer

@author Florent Forest
@version 2.0
"""

from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.initializers import GlorotUniform

from simpsom import SOMNet
from simpsom.neighborhoods import Neighborhoods


@tf.keras.saving.register_keras_serializable(package="MyLayers", name="KernelMult")
class SOMLayer(Layer):
    """
    Self-Organizing Map layer class with rectangular topology

    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` witch represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shape
        2D tensor with shape: `(n_samples, n_prototypes)`
    """

    def __init__(
        self,
        map_size,
        polygons: str = "Hexagons",
        inner_dist_type: str = "grid",
        PBC: bool = True,
        KSN: bool = True,
        **kwargs,
    ):
        if "input_shape" not in kwargs and "latent_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("latent_dim"),)
        super().__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.input_spec = InputSpec(ndim=2)
        self.prototypes = None
        self.polygons = polygons
        self.inner_dist_type = inner_dist_type
        self.PBC = PBC
        self.neighborhood = Neighborhoods(
            np, *self.map_size, self.polygons, self.inner_dist_type, self.PBC
        )
        self.nodes = np.arange(np.prod(self.map_size))
        self.distances = tf.constant(self.neighborhood.distances, dtype=tf.float32)
        self.d = 0
        self.KSN = KSN
        self.built = False

    def build(
        self,
        input_shape,
    ):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.prototypes = self.add_weight(
            shape=(self.n_prototypes, input_dim),
            initializer=GlorotUniform(),
            name="prototypes",
            trainable=True,
        )
        self.built = True

    def compute_energies(self, inputs, sigma):
        self.d = tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2
        )
        # someone else has to take care of sigma -> custom callback
        denominator = 2 * tf.pow(sigma, 2)
        self.h = tf.exp(-tf.pow(self.distances, 2) / denominator) # / tf.sqrt(denominator * np.pi)
        self.h = self.h / tf.reduce_sum(self.h, axis=1, keepdims=True)
        return self.h @ tf.transpose(self.d)

    def call(self, inputs, sigma, **kwargs):
        # Heskes 1999, Ferles et al. 2018
        energies = self.compute_energies(inputs, sigma)
        # if self.KSN:
        #     bmus = tf.math.argmin(energies, axis=0)
            
        #     D_data = tf.math.reduce_euclidean_norm(
        #         tf.expand_dims(inputs, axis=1) - tf.expand_dims(inputs, axis=0), axis=-1
        #     )
        #     D_data = D_data / tf.reduce_max(D_data)
        #     D_SOM = tf.gather(
        #         tf.gather(self.distances, indices=bmus, axis=0), indices=bmus, axis=1
        #     )
        #     D_SOM = D_SOM / tf.reduce_max(D_SOM)
        #     # l1 = tf.reduce_sum(tf.square(D_data - D_SOM), axis=1)
        #     # self.l2 = tf.shape(inputs)[0] * 10 * (tf.math.reduce_prod(self.map_size) - tf.shape(tf.unique(bmus)[0])[0])
        #     # self.l2 = tf.cast(self.l2, tf.float32)
        #     return bmus
        return 0.5 * tf.math.reduce_min(energies, axis=0)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_prototypes

    def get_bmus(self, inputs, sigma):
        energies = self.compute_energies(inputs, sigma)
        return tf.math.argmin(energies, axis=0)

    def get_config(self):
        config = {
            "map_size": self.map_size,
            "polygons": self.polygons,
            "inner_dist_type": self.inner_dist_type,
            "PBC": self.PBC,
        }
        base_config = super().get_config()
        return config | base_config
