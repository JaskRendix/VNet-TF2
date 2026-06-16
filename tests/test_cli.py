from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cli.main import main


@pytest.fixture
def fake_model():
    return MagicMock()


def test_cli_nifti(monkeypatch, fake_model):
    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)

    monkeypatch.setattr(
        "cli.main.load_nifti",
        lambda path: (np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
    )

    monkeypatch.setattr(
        "cli.main.predict",
        lambda model, vol, raw=False: np.ones((8, 8, 8, 1), dtype=np.float32),
    )

    saved = {}

    def fake_save(mask, affine, out):
        saved["mask"] = mask
        saved["affine"] = affine
        saved["out"] = out

    monkeypatch.setattr("cli.main.save_nifti", fake_save)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp_in:
        tmp_out = Path(tmp_in.name + "_out.nii.gz")

        monkeypatch.setattr(
            "sys.argv",
            ["vnet-infer", "nifti", tmp_in.name, str(tmp_out)],
        )

        main()

        assert "mask" in saved
        assert saved["out"] == str(tmp_out)


def test_cli_dicom(monkeypatch, fake_model):
    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)

    monkeypatch.setattr(
        "cli.main.load_dicom_series",
        lambda folder: np.zeros((4, 4, 4), dtype=np.float32),
    )

    monkeypatch.setattr(
        "cli.main.predict",
        lambda model, vol, raw=False: np.ones((4, 4, 4, 1), dtype=np.float32),
    )

    saved = {}
    monkeypatch.setattr(
        "cli.main.save_nifti",
        lambda mask, affine, out: saved.update({"out": out}),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "mask.nii.gz"

        monkeypatch.setattr(
            "sys.argv",
            ["vnet-infer", "dicom", tmp_dir, str(out_path)],
        )

        main()

        assert saved["out"] == str(out_path)


def test_cli_no_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vnet-infer"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


def test_cli_unknown_command(monkeypatch):
    monkeypatch.setattr("sys.argv", ["vnet-infer", "foo"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2


def test_cli_nifti_load_failure(monkeypatch, fake_model):
    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)

    monkeypatch.setattr(
        "cli.main.load_nifti",
        lambda path: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    monkeypatch.setattr(
        "sys.argv",
        ["vnet-infer", "nifti", "bad.nii.gz", "out.nii.gz"],
    )

    with pytest.raises(RuntimeError):
        main()


def test_cli_nifti_print(monkeypatch, fake_model, capsys):
    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)

    monkeypatch.setattr(
        "cli.main.load_nifti",
        lambda path: (np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
    )

    monkeypatch.setattr(
        "cli.main.predict",
        lambda model, vol, raw=False: np.ones((8, 8, 8, 1), dtype=np.float32),
    )

    monkeypatch.setattr("cli.main.save_nifti", lambda *a, **k: None)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp_in:
        tmp_out = Path(tmp_in.name + "_out.nii.gz")

        monkeypatch.setattr(
            "sys.argv",
            ["vnet-infer", "nifti", tmp_in.name, str(tmp_out)],
        )

        main()

        out = capsys.readouterr().out
        assert "[OK] Saved mask to" in out


def test_cli_model_loaded_with_dynamic_shape(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "cli.main.load_vnet",
        lambda **kwargs: captured.update(kwargs) or MagicMock(),
    )

    monkeypatch.setattr(
        "cli.main.load_nifti",
        lambda p: (np.zeros((8, 8, 8)), np.eye(4)),
    )

    monkeypatch.setattr(
        "cli.main.predict",
        lambda m, v, raw=False: np.zeros((8, 8, 8, 1)),
    )

    monkeypatch.setattr("cli.main.save_nifti", lambda *a, **k: None)

    monkeypatch.setattr(
        "sys.argv",
        ["vnet-infer", "nifti", "a.nii.gz", "b.nii.gz"],
    )

    main()

    assert captured["input_shape"] == (None, None, None, 1)


def test_cli_uses_weights(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "cli.main.load_vnet",
        lambda **kwargs: captured.update(kwargs) or MagicMock(),
    )

    monkeypatch.setattr(
        "cli.main.load_nifti", lambda p: (np.zeros((4, 4, 4)), np.eye(4))
    )
    monkeypatch.setattr(
        "cli.main.predict",
        lambda m, v, raw=False: np.zeros((4, 4, 4, 1)),
    )
    monkeypatch.setattr("cli.main.save_nifti", lambda *a, **k: None)

    monkeypatch.setattr(
        "sys.argv",
        ["vnet-infer", "--weights", "model.h5", "nifti", "a.nii.gz", "b.nii.gz"],
    )

    main()

    assert captured["weights_path"] == "model.h5"


def test_cli_classes(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "cli.main.load_vnet",
        lambda **kwargs: captured.update(kwargs) or MagicMock(),
    )

    monkeypatch.setattr(
        "cli.main.load_nifti", lambda p: (np.zeros((4, 4, 4)), np.eye(4))
    )
    monkeypatch.setattr(
        "cli.main.predict",
        lambda m, v, raw=False: np.zeros((4, 4, 4, 3)),
    )
    monkeypatch.setattr("cli.main.save_nifti", lambda *a, **k: None)

    monkeypatch.setattr(
        "sys.argv",
        ["vnet-infer", "--classes", "3", "nifti", "a.nii.gz", "b.nii.gz"],
    )

    main()

    assert captured["num_classes"] == 3


def test_cli_no_activation(monkeypatch, fake_model):
    received = {}

    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)
    monkeypatch.setattr(
        "cli.main.load_nifti", lambda p: (np.zeros((4, 4, 4)), np.eye(4))
    )

    def fake_predict(model, vol, raw=False):
        received["raw"] = raw
        return np.zeros((4, 4, 4, 1))

    monkeypatch.setattr("cli.main.predict", fake_predict)
    monkeypatch.setattr("cli.main.save_nifti", lambda *a, **k: None)

    monkeypatch.setattr(
        "sys.argv",
        ["vnet-infer", "--no-activation", "nifti", "a.nii.gz", "b.nii.gz"],
    )

    main()

    assert received["raw"] is True
