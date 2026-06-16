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


def test_preprocess_constant_volume():
    vol = np.ones((16, 16, 16), dtype=np.float32)
    out = preprocess_volume(vol)

    assert out.dtype == np.float32
    assert out.shape == (16, 16, 16, 1)
    assert not np.isnan(out).any()
    assert np.all(out == 0.0)


def test_preprocess_with_channel_dimension():
    vol = np.random.rand(16, 16, 16, 1).astype(np.float32)
    out = preprocess_volume(vol)

    assert out.shape == (16, 16, 16, 1)
    assert out.dtype == np.float32


def test_predict_without_activation():
    model = load_vnet(
        weights_path=None,
        input_shape=(16, 16, 16, 1),
        num_classes=1,
    )

    vol = np.random.rand(16, 16, 16).astype(np.float32)
    logits = predict(model, vol, apply_activation=False)

    assert logits.shape == (16, 16, 16, 1)
    assert logits.dtype == np.float32


def test_predict_multiclass_output_shape():
    model = load_vnet(
        weights_path=None,
        input_shape=(16, 16, 16, 1),
        num_classes=3,
    )

    vol = np.random.rand(16, 16, 16).astype(np.float32)
    probs = predict(model, vol)

    assert probs.shape == (16, 16, 16, 3)
    assert not np.isnan(probs).any()


def test_predict_non_cubic_volume():
    model = load_vnet(
        weights_path=None,
        input_shape=(None, None, None, 1),
        num_classes=1,
    )

    vol = np.random.rand(32, 48, 20).astype(np.float32)
    mask = predict(model, vol)

    assert mask.shape == (32, 48, 20, 1)
    assert not np.isnan(mask).any()
