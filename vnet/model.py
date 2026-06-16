from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


class ConvBlock(layers.Layer):
    def __init__(
        self,
        filters: int,
        num_convs: int,
        activation: str = "relu",
        dropout: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.filters = filters
        self.num_convs = num_convs
        self.activation = activation
        self.dropout = dropout

        self.convs: list[layers.Conv3D] = []
        for i in range(num_convs):
            self.convs.append(
                layers.Conv3D(
                    filters,
                    kernel_size=5,
                    padding="same",
                    name=f"{name}_conv{i+1}" if name is not None else None,
                )
            )

        self.shortcut_proj = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_shortcut_proj" if name is not None else None,
        )

        self.act = layers.Activation(activation)
        self.drop = layers.Dropout(dropout) if dropout > 0 else None

    def build(self, input_shape: tf.TensorShape) -> None:
        # All sublayers are already created in __init__, just mark as built
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shortcut = x

        # Project shortcut if needed
        if shortcut.shape[-1] != self.filters:
            shortcut = self.shortcut_proj(shortcut)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i == self.num_convs - 1:
                x = layers.Add()([x, shortcut])
            x = self.act(x)
            if self.drop is not None and training:
                x = self.drop(x)

        return x


class ConvBlock2(layers.Layer):
    def __init__(
        self,
        filters: int,
        num_convs: int,
        activation: str = "relu",
        dropout: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.filters = filters
        self.num_convs = num_convs
        self.activation = activation
        self.dropout = dropout

        self.concat = layers.Concatenate(axis=-1)

        self.convs: list[layers.Conv3D] = []
        for i in range(num_convs):
            self.convs.append(
                layers.Conv3D(
                    filters,
                    kernel_size=5,
                    padding="same",
                    name=f"{name}_conv{i+1}" if name is not None else None,
                )
            )

        self.shortcut_proj = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_shortcut_proj" if name is not None else None,
        )

        self.act = layers.Activation(activation)
        self.drop = layers.Dropout(dropout) if dropout > 0 else None

    def build(self, input_shape: tf.TensorShape) -> None:
        # input_shape is for the concatenated tensor; sublayers already exist
        super().build(input_shape)

    def call(self, x: tf.Tensor, skip: tf.Tensor, training: bool = False) -> tf.Tensor:
        x_shape = tf.shape(x)
        skip_shape = tf.shape(skip)

        sx = tf.minimum(x_shape[1], skip_shape[1])
        sy = tf.minimum(x_shape[2], skip_shape[2])
        sz = tf.minimum(x_shape[3], skip_shape[3])

        x = x[:, :sx, :sy, :sz, :]
        skip = skip[:, :sx, :sy, :sz, :]

        x = self.concat([x, skip])
        shortcut = self.shortcut_proj(x)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i == self.num_convs - 1:
                x = layers.Add()([x, shortcut])
            x = self.act(x)
            if self.drop is not None and training:
                x = self.drop(x)

        return x


class VNet(Model):
    def __init__(
        self,
        num_classes: int,
        num_channels: int = 16,
        num_levels: int = 4,
        num_convolutions: tuple[int, ...] = (1, 2, 3, 3),
        bottom_convolutions: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if num_levels != len(num_convolutions):
            raise ValueError("num_levels must match len(num_convolutions)")

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation = activation
        self.dropout = dropout

        # Input projection
        self.input_conv = layers.Conv3D(
            num_channels, kernel_size=5, padding="same", name="input_conv"
        )
        self.input_act = layers.Activation(activation)

        # Encoder
        self.enc_blocks: list[ConvBlock] = []
        self.down_blocks: list[layers.Layer] = []

        channels = num_channels
        for level in range(num_levels):
            self.enc_blocks.append(
                ConvBlock(
                    filters=channels,
                    num_convs=num_convolutions[level],
                    activation=activation,
                    dropout=dropout,
                    name=f"encoder_level{level+1}",
                )
            )
            self.down_blocks.append(
                layers.Conv3D(
                    channels * 2,
                    kernel_size=2,
                    strides=2,
                    padding="same",
                    name=f"encoder_down_level{level+1}",
                )
            )
            channels *= 2

        # Bottom
        self.bottom_block = ConvBlock(
            filters=channels,
            num_convs=bottom_convolutions,
            activation=activation,
            dropout=dropout,
            name="bottom",
        )

        # Decoder
        self.up_blocks: list[layers.Layer] = []
        self.dec_blocks: list[ConvBlock2] = []

        for level in reversed(range(num_levels)):
            self.up_blocks.append(
                layers.Conv3DTranspose(
                    self.enc_blocks[level].filters,
                    kernel_size=2,
                    strides=2,
                    padding="same",
                    name=f"decoder_up_level{level+1}",
                )
            )
            self.dec_blocks.append(
                ConvBlock2(
                    filters=self.enc_blocks[level].filters,
                    num_convs=num_convolutions[level],
                    activation=activation,
                    dropout=dropout,
                    name=f"decoder_level{level+1}",
                )
            )

        # Output
        self.output_conv = layers.Conv3D(
            num_classes, kernel_size=1, padding="same", name="output_conv"
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        # All sublayers are already created; just mark as built
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = inputs

        # Input projection
        if inputs.shape[-1] == 1:
            x = tf.tile(x, [1, 1, 1, 1, self.num_channels])
        else:
            x = self.input_conv(x)
            x = self.input_act(x)

        # Encoder
        skips: list[tf.Tensor] = []
        for enc, down in zip(self.enc_blocks, self.down_blocks):
            x = enc(x, training=training)
            skips.append(x)
            x = down(x)
            x = self.input_act(x)

        # Bottom
        x = self.bottom_block(x, training=training)

        # Decoder
        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = self.input_act(x)
            x = dec(x, skip, training=training)

        return self.output_conv(x)


def build_vnet(
    input_shape: tuple[int | None, int | None, int | None, int] = (128, 128, 128, 1),
    num_classes: int = 1,
    num_channels: int = 16,
    num_levels: int = 4,
    num_convolutions: tuple[int, ...] = (1, 2, 3, 3),
    bottom_convolutions: int = 3,
    activation: str = "relu",
    dropout: float = 0.0,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input_volume")
    model = VNet(
        num_classes=num_classes,
        num_channels=num_channels,
        num_levels=num_levels,
        num_convolutions=num_convolutions,
        bottom_convolutions=bottom_convolutions,
        activation=activation,
        dropout=dropout,
        name="VNet",
    )
    outputs = model(inputs)
    return Model(inputs=inputs, outputs=outputs, name="VNet")
