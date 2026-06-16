import io
import tempfile
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest
from httpx import ASGITransport, AsyncClient

from api.server import app


@pytest.mark.asyncio
async def test_api_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/")
        assert r.status_code == 200
        assert r.json()["model"] == "VNet-TF2"


@pytest.mark.asyncio
async def test_api_predict_nifti(monkeypatch):
    # mock predict → return a simple mask
    monkeypatch.setattr(
        "api.server.predict",
        lambda model, vol, raw=False: np.ones((*vol.shape, 1), dtype=np.float32),
    )

    vol = np.random.rand(16, 16, 16).astype("float32")
    affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(nii, tmp.name)
        tmp.seek(0)

        files = {"file": ("scan.nii.gz", tmp.read(), "application/octet-stream")}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post("/predict/nifti", files=files)

            assert r.status_code == 200
            assert r.headers["content-disposition"].startswith("attachment")


@pytest.mark.asyncio
async def test_api_predict_dicom_zip(monkeypatch):
    # mock predict
    monkeypatch.setattr(
        "api.server.predict",
        lambda model, vol, raw=False: np.ones((*vol.shape, 1), dtype=np.float32),
    )

    # create fake DICOM slices (uint16 to match metadata)
    slices = [(np.random.rand(8, 8) * 65535).astype("uint16") for _ in range(4)]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, arr in enumerate(slices):
            ds = pydicom.Dataset()

            # File meta
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta = file_meta

            # Required tags
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.Modality = "OT"
            ds.Rows, ds.Columns = arr.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = arr.tobytes()

            dicom_bytes = io.BytesIO()
            pydicom.dcmwrite(dicom_bytes, ds, enforce_file_format=True)
            dicom_bytes.seek(0)

            z.writestr(f"{i:03d}.dcm", dicom_bytes.read())

    buf.seek(0)
    files = {"file": ("dicoms.zip", buf.read(), "application/zip")}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict/dicom", files=files)

        assert r.status_code == 200
        assert r.headers["content-disposition"].startswith("attachment")


@pytest.mark.asyncio
async def test_api_predict_dicom_zip_invalid(monkeypatch):
    transport = ASGITransport(app=app)

    bad_zip = b"not a zip file"
    files = {"file": ("bad.zip", bad_zip, "application/zip")}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict/dicom", files=files)

        assert r.status_code == 400 or r.status_code == 422


@pytest.mark.asyncio
async def test_api_predict_dicom_zip_mixed(monkeypatch):
    # mock predict
    monkeypatch.setattr(
        "api.server.predict",
        lambda model, vol, raw=False: np.ones((*vol.shape, 1), dtype=np.float32),
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", b"hello")
        z.writestr("image.png", b"\x89PNG...")

    buf.seek(0)
    files = {"file": ("mixed.zip", buf.read(), "application/zip")}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/predict/dicom", files=files)

        # Either the API rejects it or returns a mask if your loader skips non‑DICOM files
        assert r.status_code in (200, 400, 422)
