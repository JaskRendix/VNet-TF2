from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def get_activation_layer(activation: str, name: str | None = None) -> layers.Layer:
    if activation.lower() == "prelu":
        return layers.PReLU(name=name)
    return layers.Activation(activation, name=name)


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

        # Pre-instantiate all conv layers
        self.convs = [
            layers.Conv3D(
                filters,
                kernel_size=5,
                padding="same",
                name=f"{name}_conv{i+1}" if name else None,
            )
            for i in range(num_convs)
        ]

        # Intermediate activations/dropouts (not applied after last conv)
        self.intermediate_acts = [
            get_activation_layer(activation, name=f"{name}_act{i+1}" if name else None)
            for i in range(num_convs - 1)
        ]
        self.intermediate_drops = [
            layers.Dropout(dropout) if dropout > 0 else None
            for _ in range(num_convs - 1)
        ]

        # Shortcut projection
        self.shortcut_proj = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_shortcut_proj" if name else None,
        )

        # Final residual + activation
        self.residual_add = layers.Add(name=f"{name}_add" if name else None)
        self.final_act = get_activation_layer(
            activation, name=f"{name}_final_act" if name else None
        )
        self.final_drop = layers.Dropout(dropout) if dropout > 0 else None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shortcut = x

        # Project shortcut if channels mismatch
        if shortcut.shape[-1] != self.filters:
            shortcut = self.shortcut_proj(shortcut)

        # Convolution chain
        for i in range(self.num_convs):
            x = self.convs[i](x)

            if i < self.num_convs - 1:
                x = self.intermediate_acts[i](x)
                if self.intermediate_drops[i] is not None:
                    x = self.intermediate_drops[i](x, training=training)

        # Residual + final activation
        x = self.residual_add([x, shortcut])
        x = self.final_act(x)
        if self.final_drop is not None:
            x = self.final_drop(x, training=training)

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

        self.concat = layers.Concatenate(
            axis=-1, name=f"{name}_concat" if name else None
        )

        # Reduce concatenated channels to target filter count
        self.channel_reduction = layers.Conv3D(
            filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_channel_reduction" if name else None,
        )

        # Reuse encoder block logic
        self.block = ConvBlock(
            filters=filters,
            num_convs=num_convs,
            activation=activation,
            dropout=dropout,
            name=f"{name}_sub_block" if name else None,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x: tf.Tensor, skip: tf.Tensor, training: bool = False) -> tf.Tensor:
        x_shape = tf.shape(x)
        skip_shape = tf.shape(skip)

        # Dynamic cropping
        sx = tf.minimum(x_shape[1], skip_shape[1])
        sy = tf.minimum(x_shape[2], skip_shape[2])
        sz = tf.minimum(x_shape[3], skip_shape[3])

        x = x[:, :sx, :sy, :sz, :]
        skip = skip[:, :sx, :sy, :sz, :]

        # Concatenate → reduce channels → residual block
        x = self.concat([x, skip])
        x = self.channel_reduction(x)
        return self.block(x, training=training)


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

        # Input projection
        self.input_conv = layers.Conv3D(
            num_channels, kernel_size=5, padding="same", name="input_conv"
        )
        self.input_act = get_activation_layer(activation, name="input_act")

        # Encoder
        self.enc_blocks = []
        self.down_blocks = []
        self.down_acts = []

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
            self.down_acts.append(
                get_activation_layer(activation, name=f"down_act_level{level+1}")
            )
            channels *= 2

        # Bottom block
        self.bottom_block = ConvBlock(
            filters=channels,
            num_convs=bottom_convolutions,
            activation=activation,
            dropout=dropout,
            name="bottom",
        )

        # Decoder
        self.up_blocks = []
        self.up_acts = []
        self.dec_blocks = []

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
            self.up_acts.append(
                get_activation_layer(activation, name=f"up_act_level{level+1}")
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

        # Output head
        self.output_conv = layers.Conv3D(
            num_classes, kernel_size=1, padding="same", name="output_conv"
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = inputs

        # Input projection
        if inputs.shape[-1] == 1:
            x = tf.tile(x, [1, 1, 1, 1, self.enc_blocks[0].filters])
        else:
            x = self.input_conv(x)
            x = self.input_act(x)

        # Encoder
        skips = []
        for enc, down, act in zip(self.enc_blocks, self.down_blocks, self.down_acts):
            x = enc(x, training=training)
            skips.append(x)
            x = down(x)
            x = act(x)

        # Bottom
        x = self.bottom_block(x, training=training)

        # Decoder
        for up, act, dec, skip in zip(
            self.up_blocks, self.up_acts, self.dec_blocks, reversed(skips)
        ):
            x = up(x)
            x = act(x)
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
    vnet = VNet(
        num_classes=num_classes,
        num_channels=num_channels,
        num_levels=num_levels,
        num_convolutions=num_convolutions,
        bottom_convolutions=bottom_convolutions,
        activation=activation,
        dropout=dropout,
        name="VNet_Core",
    )
    outputs = vnet(inputs)
    return Model(inputs=inputs, outputs=outputs, name="VNet")
