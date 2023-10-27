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
        self.polygons = polygons
        self.inner_dist_type = inner_dist_type
        self.neighborhood_fun = neighborhood_fun
        self.PBC = PBC
        self.neighborhood = Neighborhoods(
            np, *self.map_size, self.polygons, self.inner_dist_type, self.PBC
        )
        self.neighborhood_caller = partial(
            self.neighborhood.neighborhood_caller, neigh_func=self.neighborhood_fun
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
        
    def compute_energies(self, inputs):
        self.d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2) 
        # someone else has to take care of sigma -> custom callback
        self.h = tf.constant(self.neighborhood_caller(self.nodes, sigma=self.sigma), dtype=tf.float32) 
        energies = self.h @ tf.transpose(self.d)
        return energies

    def call(self, inputs, **kwargs):
        # Heskes 1999, Ferles et al. 2018
        energies = self.compute_energies(inputs)
        bmus = tf.math.argmin(energies, axis=0)
        self.add_loss(tf.reduce_mean(tf.math.abs(tf.norm(self.prototypes, axis=1) - 1)))
        return 0.5 * tf.experimental.numpy.take_along_axis(energies, tf.cast(bmus[None, :], tf.int32), axis=0)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_prototypes
    
    def get_bmus(self, inputs):
        return tf.math.argmin(tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2
        ), axis=1)

    def get_config(self):
        config = {
            'map_size': self.map_size, 
            'polygons': self.polygons, 
            'inner_dist_type': self.inner_dist_type,
            'neighborhood_fun': self.neighborhood_fun,
            'PBC': self.PBC,
        }
        base_config = super(SOMLayer, self).get_config()
        return config | base_config
