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

from simpsom import SOMNet
from simpsom.neighborhoods import Neighborhoods


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
        neighborhood_fun: str = "gaussian",
        PBC: bool = True,
        **kwargs,
    ):
        if "input_shape" not in kwargs and "latent_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("latent_dim"),)
        super().__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.input_spec = InputSpec(ndim=2)
        self.prototypes = None
        self.neighborhood = Neighborhoods(
            np, *self.map_size, polygons, inner_dist_type, PBC
        )
        self.neighborhood_caller = partial(
            self.neighborhood.neighborhood_caller, neigh_func=neighborhood_fun
        )
        self.sigma = 1 # taken care of by someone else
        self.nodes = np.arange(np.prod(self.map_size))
        self.d = 0
        self.built = False

    def build(
        self,
        input_shape,
    ):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.prototypes = self.add_weight(
            shape=(self.n_prototypes, input_dim),
            initializer="glorot_uniform",
            name="prototypes",
            trainable=True,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Calculate pairwise squared euclidean distances between inputs and prototype vectors

        Arguments:
            inputs: the variable containing data, Tensor with shape `(n_samples, latent_dim)`
        Return:
            d: distances between inputs and prototypes, Tensor with shape `(n_samples, n_prototypes)`
        """
        # Note: (tf.expand_dims(inputs, axis=1) - self.prototypes) has shape (n_samples, n_prototypes, latent_dim)
        self.d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2) 
        # someone else has to take care of sigma -> custom training step
        self.h = tf.constant(self.neighborhood_caller(self.nodes, sigma=self.sigma), dtype=tf.float32) 
        energies = self.h @ tf.transpose(self.d)
        bmus = tf.math.argmin(energies, axis=0) # Heskes 1999, Ferles et al. 2018
        
        self.add_loss(0.5 * tf.experimental.numpy.take_along_axis(energies, tf.cast(bmus[None, :], tf.int32), axis=0))
        return bmus

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_prototypes
    
    def get_bmus(self, inputs):
        return tf.math.argmin(tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2
        ), axis=1)

    def get_config(self):
        config = {"map_size": self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
