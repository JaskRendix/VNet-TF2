import numpy as np

from vnet.inference import load_vnet, predict, preprocess_volume


def test_preprocess_volume():
    vol = np.random.randint(0, 255, (16, 16, 16)).astype("uint8")
    out = preprocess_volume(vol)

    assert out.dtype == "float32"
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.shape == (16, 16, 16, 1)


def test_predict_dummy_volume():
    model = load_vnet(
        weights_path=None,
        input_shape=(16, 16, 16, 1),
        num_classes=1,
    )

    vol = np.random.rand(16, 16, 16).astype("float32")
    mask = predict(model, vol)

    assert mask.shape == (16, 16, 16, 1)
    assert not np.isnan(mask).any()
