"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper functions

@author Florent Forest
@version 2.0
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Cropping2D,
    ZeroPadding2D,
    Resizing,
    Flatten,
    Reshape,
    Normalization,
    BatchNormalization,
)
import numpy as np


def mlp_autoencoder(encoder_dims, act="relu", init="glorot_uniform", batchnorm=False):
    """Fully connected symmetric autoencoder model.

    Parameters
    ----------
    encoder_dims : list
        number of units in each layer of encoder. encoder_dims[0] is the input dim, encoder_dims[-1] is the
        size of the hidden layer (latent dim). The autoencoder is symmetric, so the total number of layers
        is 2*len(encoder_dims) - 1
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : bool (default=False)
        use batch normalization

    Returns
    -------
    ae_model, encoder_model, decoder_model : tuple
        autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name="input")
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks - 1):
        encoded = Dense(
            encoder_dims[i + 1],
            activation=act,
            kernel_initializer=init,
            name="encoder_%d" % i,
        )(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
    # Hidden layer (latent space)
    encoded = Dense(
        encoder_dims[-1],
        activation="linear",
        kernel_initializer=init,
        name="encoder_%d" % (n_stacks - 1),
    )(
        encoded
    )  # latent representation is extracted from here
    # Internal layers in decoder
    decoded = encoded
    for i in range(n_stacks - 1, 0, -1):
        decoded = Dense(
            encoder_dims[i],
            activation=act,
            kernel_initializer=init,
            name="decoder_%d" % i,
        )(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
    # Output
    decoded = Dense(
        encoder_dims[0], activation="linear", kernel_initializer=init, name="decoder_0"
    )(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name="AE")

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name="encoder")

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks - 1, -1, -1):
        decoded = autoencoder.get_layer("decoder_%d" % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name="decoder")

    return autoencoder, encoder, decoder


def conv2d_autoencoder(
    input_shape: list | tuple,
    latent_dim: int,
    encoder_filters: list | tuple,
    filter_size: int = 3,
    pooling_size: int = 2,
    strides: int = 1,
    unequal_strat: str = 'stretch',
    act="relu",
    batchnorm=False,
):
    """2D convolutional autoencoder model.

    Parameters
    ----------
    input_shape : tuple
        input shape given as (height, width, channels) tuple
    latent_dim : int
        dimension of latent code (units in hidden dense layer)
    encoder_filters : list
        number of filters in each layer of encoder. The autoencoder is symmetric,
        so the total number of layers is 2*len(encoder_filters) - 1
    filter_size : int
        size of conv filters
    pooling_size : int
        size of maxpool filters
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : boolean (default=False)
        use batch normalization

    Returns
    -------
        ae_model, encoder_model, decoder_model : tuple
            autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_filters)
    x = Input(shape=input_shape, name="input")

    # stretching
    if input_shape[0] != input_shape[1]:
        smaller, bigger = np.argsort(input_shape[:2])
        divisible_by = strides ** len(encoder_filters)
        factor = int(np.ceil(input_shape[bigger] / divisible_by))
        new_shape = (divisible_by * factor, divisible_by * factor, 1)
        if unequal_strat == 'stretch':
            encoded = Resizing(*new_shape[:2], name='resize_input')(x)
        elif unequal_strat == 'pad':
            padding = [None, None]
            padding[bigger] = (0, 0)
            pad_size = input_shape[bigger] - input_shape[smaller]
            padding[smaller] = (pad_size // 2, int(np.ceil(pad_size / 2)))
            encoded = ZeroPadding2D(padding, name='pad_input')(x)
        else:
            raise NotImplementedError('Only stretch unequal strat for now')
    else:
        encoded = x
        new_shape = input_shape
    # Infer code shape (assuming "same" padding, conv stride equal to 1 and max pooling stride equal to pooling_size)
    # Input
    # Internal layers in encoder
    for i in range(n_stacks):
        encoded = Conv2D(
            encoder_filters[i],
            filter_size,
            activation=act,
            strides=strides,
            padding="same",
            name="encoder_conv_%d" % i,
        )(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
        # encoded = MaxPooling2D(
        #     pooling_size, padding="same", name="encoder_maxpool_%d" % i
        # )(encoded)
    code_shape = encoded.shape[1:]
    # Flatten
    flattened = Flatten(name="flatten")(encoded)
    # Project using dense layer
    code = Dense(latent_dim, name="dense1")(
        flattened
    )  # latent representation is extracted from here
    
    # encoder model
    encoder = Model(inputs=x, outputs=code, name="encoder")
    
    latent_input = Input(shape=(latent_dim,))
    reshaped = Dense(code_shape[0] * code_shape[1] * code_shape[2], name="dense2")(latent_input)
    # Reshape
    reshaped = Reshape(code_shape, name="reshape")(reshaped)
    # Internal layers in decoder
    decoded = reshaped
    for i in range(n_stacks - 1, -1, -1):
        decoded = Conv2DTranspose(
            encoder_filters[i],
            filter_size,
            activation=act,
            strides=strides,
            padding="same",
            name="decoder_conv_%d" % i,
        )(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
        # decoded = UpSampling2D(pooling_size, name="decoder_upsample_%d" % i)(decoded)
    # Output
    decoded = Conv2DTranspose(
        1, filter_size, activation="sigmoid", padding="same", strides=1, name="decoder_0"
    )(decoded)
    
    if input_shape[0] != input_shape[1]:
        if unequal_strat == 'stretch':
            decoded = Resizing(*input_shape[:2], name='resize_output')(decoded)
        elif unequal_strat == 'pad':
            decoded = Cropping2D(padding, name='crop_output')(decoded)
    # decoder model
    decoder = Model(inputs=latent_input, outputs=decoded, name="decoder")

    return encoder, decoder
