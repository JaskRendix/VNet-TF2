import numpy as np
import pydicom

from vnet.inference import (
    _sigmoid_numpy,
    _softmax_numpy,
    load_dicom_series,
    load_vnet,
    predict,
    preprocess_volume,
)


class FakeDicom:
    def __init__(self, array, ipp=None, instance=None, frames=None):
        self._arr = array
        if ipp is not None:
            self.ImagePositionPatient = ipp
        if instance is not None:
            self.InstanceNumber = instance
        if frames is not None:
            self.NumberOfFrames = frames

    @property
    def pixel_array(self):
        return self._arr


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


def test_numpy_sigmoid_stability():
    x = np.array([-1000, 0, 1000], dtype=np.float32)
    y = _sigmoid_numpy(x)

    assert y.dtype == np.float32
    assert y[0] >= 0.0  # allow 0.0 due to float32 underflow
    assert y[1] == 0.5
    assert y[2] <= 1.0
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_numpy_softmax_properties():
    x = np.random.randn(4, 4, 4, 3).astype(np.float32)
    y = _softmax_numpy(x, axis=-1)

    assert y.shape == x.shape
    assert np.allclose(np.sum(y, axis=-1), 1.0, atol=1e-5)
    assert not np.isnan(y).any()


def test_preprocess_without_normalization():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    out = preprocess_volume(vol, normalize=False)

    assert np.allclose(out[..., 0], vol)
    assert out.dtype == np.float32
    assert out.shape == (8, 8, 8, 1)


def test_predict_without_preprocess():
    model = load_vnet(
        weights_path=None,
        input_shape=(8, 8, 8, 1),
        num_classes=1,
    )

    vol = np.random.rand(8, 8, 8).astype(np.float32)
    logits = predict(model, vol, preprocess=False)

    assert logits.shape == (8, 8, 8, 1)
    assert not np.isnan(logits).any()


def test_dicom_loader_multiframe(tmp_path):
    arr = np.random.rand(5, 32, 32).astype(np.float32)
    ds = FakeDicom(arr, frames=5)

    def fake_dcmread(path):
        return ds

    pydicom.dcmread = fake_dcmread

    (tmp_path / "frame.dcm").write_bytes(b"dummy")

    vol = load_dicom_series(tmp_path)

    assert vol.shape == (5, 32, 32)
    assert vol.dtype == np.float32


def test_dicom_loader_stable_sorting(tmp_path):
    slices = []
    for i, name in enumerate(["b.dcm", "a.dcm", "c.dcm"]):
        arr = np.full((16, 16), i, dtype=np.float32)
        ds = FakeDicom(arr, ipp=[0, 0, 10.0])
        slices.append((name, ds))

    def fake_dcmread(path):
        for name, ds in slices:
            if path.name == name:
                return ds

    pydicom.dcmread = fake_dcmread

    for name, _ in slices:
        (tmp_path / name).write_bytes(b"dummy")

    vol = load_dicom_series(tmp_path)

    assert vol[0, 0, 0] == 1  # a.dcm
    assert vol[1, 0, 0] == 0  # b.dcm
    assert vol[2, 0, 0] == 2  # c.dcm


def test_dicom_loader_missing_metadata(tmp_path):
    slices = []
    for i in range(3):
        arr = np.full((8, 8), i, dtype=np.float32)
        ds = FakeDicom(arr)
        slices.append((f"{i}.dcm", ds))

    def fake_dcmread(path):
        for name, ds in slices:
            if path.name == name:
                return ds

    pydicom.dcmread = fake_dcmread

    for name, _ in slices:
        (tmp_path / name).write_bytes(b"dummy")

    vol = load_dicom_series(tmp_path)

    assert vol[0, 0, 0] == 0
    assert vol[1, 0, 0] == 1
    assert vol[2, 0, 0] == 2
