"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper functions

@author Florent Forest
@version 2.0
"""

import tensorflow as tf
from keras import activations, initializers, regularizers, constraints, backend
from keras.models import Model
from keras.dtensor import utils
from keras.utils import conv_utils
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
    Layer,
    InputSpec,
)


class Conv2DTransposeTied(Conv2DTranspose):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers or `None`, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
      output_padding: An integer or tuple/list of 2 integers,
        specifying the amount of padding along the height and width
        of the output tensor.
        Can be a single integer to specify the same value for all
        spatial dimensions.
        The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.
        If set to `None` (default), the output shape is inferred.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        When unspecified, uses `image_data_format` value found in your Keras
        config file at `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to "channels_last".
      dilation_rate: an integer, specifying the dilation rate for all spatial
        dimensions for dilated convolution. Specifying different dilation rates
        for different dimensions is not supported.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix
        (see `keras.initializers`). Defaults to 'glorot_uniform'.
      bias_initializer: Initializer for the bias vector
        (see `keras.initializers`). Defaults to 'zeros'.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation") (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix
        (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector
        (see `keras.constraints`).

    Input shape:
      4D tensor with shape:
      `(batch_size, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `(batch_size, filters, new_rows, new_cols)` if
      data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, new_rows, new_cols, filters)` if
      data_format='channels_last'.  `rows` and `cols` values might have changed
      due to padding.
      If `output_padding` is specified:
      ```
      new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
      output_padding[0])
      new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
      output_padding[1])
      ```

    Returns:
      A tensor of rank 4 representing
      `activation(conv2dtranspose(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References:
      - [A guide to convolution arithmetic for deep
        learning](https://arxiv.org/abs/1603.07285v1)
      - [Deconvolutional
        Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    @utils.allow_initializer_layout
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        tied_to=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs,
        )

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, "output_padding", allow_zero=True
            )
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError(
                        "Strides must be greater than output padding. "
                        f"Received strides={self.strides}, "
                        f"output_padding={self.output_padding}."
                    )
        self.tied_to = tied_to

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                "Inputs should have rank 4. " f"Received input_shape={input_shape}."
            )
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "to `Conv2DTranspose` should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        if self.tied_to is not None:
            self.kernel = self.tied_to.kernel
            self._non_trainable_weights.append(self.kernel)

        else:
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        self.kernel = self.tied_to.kernel
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        # Use the constant height and weight when possible.
        # TODO(scottzhu): Extract this into a utility function that can be
        # applied to all convolutional layers, which currently lost the static
        # shape information due to tf.shape().
        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if not tf.executing_eagerly() and inputs.shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == "channels_first":
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config["output_padding"] = self.output_padding
        return config


class DenseTied(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        tied_to=None,
        **kwargs,
    ):
        self.tied_to = tied_to
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] == self.units
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        self.kernel = tf.transpose(self.tied_to.kernel)
        output = backend.dot(inputs, self.kernel)
        if self.use_bias:
            output = backend.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output


def conv2d_autoencoder(
    input_shape: list | tuple,
    latent_dim: int,
    encoder_filters: list | tuple,
    filter_size: int | list | tuple = 3,
    strides: int | list | tuple = 1,
    pooling_size: int | list | tuple = 1,
    act="relu",
    batchnorm=False,
    tied_weights=False,
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

    if isinstance(strides, int):
        strides = [strides] * len(encoder_filters)

    if isinstance(filter_size, int):
        filter_size = [filter_size] * len(encoder_filters)

    if isinstance(pooling_size, int):
        pooling_size = [pooling_size] * len(encoder_filters)

    encoded = x
    # Input
    # Internal layers in encoder
    conv_layers = []  # store in list in case of tied weights
    for i in range(n_stacks):
        conv_layers.append(
            Conv2D(
                encoder_filters[i],
                filter_size[i],
                activation=act,
                strides=strides[i],
                padding="same",
                name=f"encoder_conv_{i}",
            )
        )
        encoded = conv_layers[-1](encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D(
            pooling_size[i], padding="same", name="encoder_maxpool_%d" % i
        )(encoded)
    code_shape = encoded.shape[1:]
    # Flatten
    flattened = Flatten(name="flatten")(encoded)
    # Project using dense layer
    final_dense = Dense(latent_dim, name="dense1")
    code = final_dense(flattened)  # latent representation is extracted from here

    # encoder model
    encoder = Model(inputs=x, outputs=code, name="encoder")

    latent_input = Input(shape=(latent_dim,))
    if tied_weights:
        reshaped = DenseTied(
            code_shape[0] * code_shape[1] * code_shape[2],
            name="dense2",
            tied_to=final_dense,
        )(latent_input)
    else:
        reshaped = Dense(code_shape[0] * code_shape[1] * code_shape[2], name="dense2")(
            latent_input
        )
    # Reshape
    reshaped = Reshape(code_shape, name="reshape")(reshaped)
    # Internal layers in decoder
    decoded = reshaped
    for i in range(n_stacks - 1, -1, -1):
        if tied_weights and i < (n_stacks - 1):
            decoded = Conv2DTransposeTied(
                encoder_filters[i],
                filter_size[i],
                activation=act,
                strides=strides[i],
                padding="same",
                name=f"decoder_conv_{i}",
                tied_to=conv_layers[i + 1],
            )(decoded)
        else:
            decoded = Conv2DTranspose(
                encoder_filters[i],
                filter_size[i],
                activation=act,
                strides=strides[i],
                padding="same",
                name=f"decoder_conv_{i}",
            )(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
        decoded = UpSampling2D(pooling_size[i], name="decoder_upsample_%d" % i)(decoded)
    # Output
    if tied_weights:
        decoded = Conv2DTransposeTied(
            1,
            filter_size[0],
            activation="sigmoid",
            strides=1,
            padding="same",
            name=f"decoder_0",
            tied_to=conv_layers[0],
        )(decoded)
    else:
        decoded = Conv2DTranspose(
            1,
            filter_size[0],
            activation="sigmoid",
            padding="same",
            strides=1,
            name="decoder_0",
        )(decoded)

    # decoder model
    decoder = Model(inputs=latent_input, outputs=decoded, name="decoder")

    return encoder, decoder
