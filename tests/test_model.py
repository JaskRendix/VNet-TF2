import numpy as np

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
