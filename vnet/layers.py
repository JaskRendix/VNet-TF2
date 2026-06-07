import tensorflow as tf
from tensorflow.keras import layers


def conv3d(
    filters: int,
    kernel_size: int = 5,
    activation: str = "relu",
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.keras.Sequential:
    """
    Single Conv3D → Activation → optional Dropout.
    """
    ops = [
        layers.Conv3D(filters, kernel_size, padding="same", name=name),
        layers.Activation(activation),
    ]
    if dropout > 0.0:
        ops.append(layers.Dropout(dropout))
    return tf.keras.Sequential(ops)


def conv_block(
    x: tf.Tensor,
    filters: int,
    num_convs: int,
    activation: str = "relu",
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:
    """
    Encoder-style residual block.
    Residual is applied only on the last convolution.
    """
    shortcut = x

    # Project shortcut if channels do not match
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=None if name is None else f"{name}_shortcut_proj",
        )(shortcut)

    for i in range(num_convs):
        x = layers.Conv3D(
            filters,
            kernel_size=5,
            padding="same",
            name=None if name is None else f"{name}_conv{i+1}",
        )(x)

        if i == num_convs - 1:
            x = layers.Add(name=None if name is None else f"{name}_residual")(
                [x, shortcut]
            )

        x = layers.Activation(activation)(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)

    return x


def conv_block_2(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    num_convs: int,
    activation: str = "relu",
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:
    """
    Decoder-style residual block.
    Concatenates skip connection, then applies convolutions.
    Residual is applied only on the last convolution.
    """
    x = layers.Concatenate(axis=-1, name=None if name is None else f"{name}_concat")(
        [x, skip]
    )

    shortcut = x

    # Project shortcut to match conv output channels
    shortcut = layers.Conv3D(
        filters,
        kernel_size=1,
        padding="same",
        name=None if name is None else f"{name}_shortcut_proj",
    )(shortcut)

    for i in range(num_convs):
        x = layers.Conv3D(
            filters,
            kernel_size=5,
            padding="same",
            name=None if name is None else f"{name}_conv{i+1}",
        )(x)

        if i == num_convs - 1:
            x = layers.Add(name=None if name is None else f"{name}_residual")(
                [x, shortcut]
            )

        x = layers.Activation(activation)(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)

    return x


def downsample(filters: int, name: str | None = None) -> layers.Layer:
    """
    Downsampling via strided Conv3D.
    """
    return layers.Conv3D(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
        name=name,
    )


def upsample(filters: int, name: str | None = None) -> layers.Layer:
    """
    Upsampling via Conv3DTranspose.
    """
    return layers.Conv3DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
        name=name,
    )
