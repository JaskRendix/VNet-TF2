import tempfile

import nibabel as nib
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from api.server import app


@pytest.mark.asyncio
async def test_api_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/")
        assert r.status_code == 200
        assert "model" in r.json()


@pytest.mark.asyncio
async def test_api_predict_nifti():
    vol = np.random.rand(16, 16, 16).astype("float32")
    affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine)

    # write to a real temporary file
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(nii, tmp.name)
        tmp.seek(0)

        files = {"file": ("scan.nii.gz", tmp.read(), "application/octet-stream")}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            r = await ac.post("/predict/nifti", files=files)

            assert r.status_code == 200
            assert r.headers["content-disposition"].startswith("attachment")
