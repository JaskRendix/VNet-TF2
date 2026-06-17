import pytest
import tensorflow as tf

from vnet.layers import conv_block, conv_block_2, downsample, upsample


@pytest.mark.parametrize(
    "filters,num_convs",
    [
        (8, 1),
        (16, 2),
        (32, 3),
    ],
)
def test_conv_block_shapes(filters, num_convs):
    x = tf.random.uniform((1, 20, 24, 28, 4))
    y = conv_block(x, filters=filters, num_convs=num_convs)
    assert y.shape == (1, 20, 24, 28, filters)


def test_conv_block_channel_projection():
    x = tf.random.uniform((1, 16, 16, 16, 3))  # channels != filters
    y = conv_block(x, filters=8, num_convs=2)
    assert y.shape == (1, 16, 16, 16, 8)


def test_conv_block_2_dynamic_crop():
    x = tf.random.uniform((1, 16, 20, 12, 32))
    skip = tf.random.uniform((1, 15, 18, 10, 32))  # smaller spatial dims
    y = conv_block_2(x, skip, filters=16, num_convs=2)
    assert y.shape == (1, 15, 18, 10, 16)


@pytest.mark.parametrize(
    "shape_x,shape_skip",
    [
        ((16, 16, 16), (16, 16, 16)),
        ((20, 30, 10), (18, 28, 9)),
        ((32, 48, 20), (30, 46, 18)),
    ],
)
def test_conv_block_2_param_shapes(shape_x, shape_skip):
    x = tf.random.uniform((1, *shape_x, 8))
    skip = tf.random.uniform((1, *shape_skip, 8))
    y = conv_block_2(x, skip, filters=8, num_convs=2)

    sx = min(shape_x[0], shape_skip[0])
    sy = min(shape_x[1], shape_skip[1])
    sz = min(shape_x[2], shape_skip[2])

    assert y.shape == (1, sx, sy, sz, 8)


def test_conv_block_2_residual():
    x = tf.random.uniform((1, 16, 16, 16, 8))
    skip = tf.random.uniform((1, 16, 16, 16, 8))
    y = conv_block_2(x, skip, filters=8, num_convs=1)
    assert y.shape == (1, 16, 16, 16, 8)


def test_downsample_forward():
    layer = downsample(filters=16)
    x = tf.random.uniform((1, 32, 32, 32, 8))
    y = layer(x)
    assert y.shape == (1, 16, 16, 16, 16)


def test_upsample_forward():
    layer = upsample(filters=8)
    x = tf.random.uniform((1, 16, 16, 16, 8))
    y = layer(x)
    assert y.shape == (1, 32, 32, 32, 8)


def test_down_up_roundtrip():
    down = downsample(filters=16)
    up = upsample(filters=8)

    x = tf.random.uniform((1, 32, 32, 32, 8))
    y = down(x)
    z = up(y)

    assert z.shape == (1, 32, 32, 32, 8)


def test_prelu_shared_axes():
    x = tf.random.uniform((1, 16, 16, 16, 8))
    y = conv_block(x, filters=8, num_convs=2, activation="prelu")
    assert y.shape == (1, 16, 16, 16, 8)


def test_conv_block_residual_addition():
    x = tf.random.uniform((1, 16, 16, 16, 8))
    y = conv_block(x, filters=8, num_convs=2)

    # Residual block must preserve shape
    assert y.shape == x.shape


def test_conv_block_2_channel_reduction():
    x = tf.random.uniform((1, 16, 16, 16, 32))
    skip = tf.random.uniform((1, 16, 16, 16, 32))

    y = conv_block_2(x, skip, filters=16, num_convs=2)

    assert y.shape == (1, 16, 16, 16, 16)


def test_conv_block_2_cropping():
    x = tf.random.uniform((1, 20, 30, 40, 16))
    skip = tf.random.uniform((1, 18, 28, 38, 16))

    y = conv_block_2(x, skip, filters=16, num_convs=2)

    assert y.shape == (1, 18, 28, 38, 16)
