import numpy as np
import tensorflow as tf

from vnet.model import build_vnet


def test_vnet_forward_pass():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 1)
    assert not np.isnan(y.numpy()).any()


def test_vnet_forward_odd_dimensions():
    model = build_vnet(
        input_shape=(33, 41, 27, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 33, 41, 27, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 33, 41, 27, 1)
    assert not np.isnan(y.numpy()).any()


def test_vnet_forward_small_volume():
    model = build_vnet(
        input_shape=(8, 8, 8, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 8, 8, 8, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 8, 8, 8, 1)


def test_vnet_forward_multiclass():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=4,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 4)


def test_vnet_forward_batch_two():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
    )

    x = np.random.rand(2, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (2, 32, 32, 32, 1)


def test_vnet_forward_channels_gt1():
    model = build_vnet(
        input_shape=(32, 32, 32, 3),
        num_classes=1,
    )

    x = np.random.rand(1, 32, 32, 32, 3).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_forward_dynamic_shape_cropping():
    model = build_vnet(
        input_shape=(32, 48, 20, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 32, 48, 20, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 48, 20, 1)


def test_vnet_bottom_block_shape():
    model = build_vnet(
        input_shape=(64, 64, 64, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 64, 64, 64, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 64, 64, 64, 1)


def test_vnet_asymmetric_large_difference():
    model = build_vnet(
        input_shape=(16, 64, 24, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 16, 64, 24, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 16, 64, 24, 1)


def test_vnet_skip_cropping_effective():
    model = build_vnet(
        input_shape=(30, 50, 22, 1),
        num_classes=1,
    )

    x = np.random.rand(1, 30, 50, 22, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 30, 50, 22, 1)


def test_vnet_with_dropout():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
        dropout=0.2,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x, training=True)

    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_custom_convolution_depths():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
        num_convolutions=(2, 2, 2, 2),
        bottom_convolutions=4,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_more_channels():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
        num_channels=32,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_odd_channel_count():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
        num_channels=12,
    )

    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_prelu_forward():
    model = build_vnet(
        input_shape=(32, 32, 32, 1),
        num_classes=1,
        activation="prelu",
    )
    x = np.random.rand(1, 32, 32, 32, 1).astype("float32")
    y = model(x)
    assert y.shape == (1, 32, 32, 32, 1)


def test_vnet_backward_pass():
    model = build_vnet(input_shape=(16, 16, 16, 1))
    x = tf.random.uniform((1, 16, 16, 16, 1))
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None for g in grads)
