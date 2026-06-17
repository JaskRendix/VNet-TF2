from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _apply_activation(x, activation: str, name: str | None):
    if activation.lower() == "prelu":
        return layers.PReLU(shared_axes=[1, 2, 3], name=name)(x)
    return layers.Activation(activation, name=name)(x)


def conv_block(
    x: tf.Tensor,
    filters: int,
    num_convs: int,
    activation: str = "prelu",
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:
    """
    V-Net encoder residual block:
    - N convolutions
    - Add shortcut BEFORE final activation
    - Optional dropout
    """
    shortcut = x

    # Project shortcut if channels mismatch
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=None if name is None else f"{name}_shortcut_proj",
        )(shortcut)

    # Convolution chain
    for i in range(num_convs):
        x = layers.Conv3D(
            filters,
            kernel_size=5,
            padding="same",
            name=None if name is None else f"{name}_conv{i+1}",
        )(x)

        # Intermediate activations (not after last conv)
        if i < num_convs - 1:
            x = _apply_activation(
                x,
                activation,
                name=None if name is None else f"{name}_act{i+1}",
            )
            if dropout > 0.0:
                x = layers.Dropout(
                    dropout,
                    name=None if name is None else f"{name}_drop{i+1}",
                )(x)

    # Residual addition BEFORE final activation
    x = layers.Add(name=None if name is None else f"{name}_residual")([x, shortcut])

    # Final activation
    x = _apply_activation(
        x,
        activation,
        name=None if name is None else f"{name}_final_act",
    )

    if dropout > 0.0:
        x = layers.Dropout(
            dropout,
            name=None if name is None else f"{name}_final_drop",
        )(x)

    return x


def conv_block_2(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    num_convs: int,
    activation: str = "prelu",
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:
    """
    V-Net decoder block:
    - Align spatial dims
    - Concatenate skip
    - Reduce channels to target filters
    - Apply encoder-style residual block
    """
    # Dynamic cropping to match shapes
    x_shape = tf.shape(x)
    skip_shape = tf.shape(skip)

    sx = tf.minimum(x_shape[1], skip_shape[1])
    sy = tf.minimum(x_shape[2], skip_shape[2])
    sz = tf.minimum(x_shape[3], skip_shape[3])

    x = x[:, :sx, :sy, :sz, :]
    skip = skip[:, :sx, :sy, :sz, :]

    # Concatenate along channels
    x = layers.Concatenate(
        axis=-1,
        name=None if name is None else f"{name}_concat",
    )([x, skip])

    # Reduce channels to target filter count
    x = layers.Conv3D(
        filters,
        kernel_size=1,
        padding="same",
        name=None if name is None else f"{name}_reduce_channels",
    )(x)

    # Reuse encoder block logic
    return conv_block(
        x=x,
        filters=filters,
        num_convs=num_convs,
        activation=activation,
        dropout=dropout,
        name=name,
    )


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
