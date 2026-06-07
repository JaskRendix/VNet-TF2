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
    # Mock load_vnet → returns fake model
    monkeypatch.setattr("cli.main.load_vnet", lambda **kwargs: fake_model)

    # Mock load_nifti → returns (volume, affine)
    monkeypatch.setattr(
        "cli.main.load_nifti",
        lambda path: (np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
    )

    # Mock predict → returns mask
    monkeypatch.setattr(
        "cli.main.predict",
        lambda model, vol: np.ones((8, 8, 8, 1), dtype=np.float32),
    )

    # Mock save_nifti
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
        lambda model, vol: np.ones((4, 4, 4, 1), dtype=np.float32),
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
