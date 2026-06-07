import io
import tempfile
import zipfile

import nibabel as nib
import numpy as np
import pydicom
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vnet.inference import load_nifti, load_vnet, predict

app = FastAPI(title="VNet Inference API", version="1.0.0")


MODEL = load_vnet(
    weights_path=None,
    input_shape=(None, None, None, 1),
    num_classes=1,
)


class PredictResponse(BaseModel):
    message: str
    mask_shape: tuple[int, ...]


def load_dicom_zip(file: UploadFile) -> np.ndarray:
    """Extract a ZIP of DICOM slices into a 3D numpy volume."""
    data = file.file.read()
    z = zipfile.ZipFile(io.BytesIO(data))

    slices = []
    for name in sorted(z.namelist()):
        if name.lower().endswith(".dcm"):
            ds = pydicom.dcmread(io.BytesIO(z.read(name)))
            slices.append(ds.pixel_array)

    volume = np.stack(slices, axis=0).astype(np.float32)
    return volume


def nifti_to_bytes(mask: np.ndarray, affine: np.ndarray) -> io.BytesIO:
    """Convert mask to NIfTI bytes for download."""
    nii = nib.Nifti1Image(mask.astype(np.float32), affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(nii, tmp.name)
        tmp.seek(0)
        data = tmp.read()

    return io.BytesIO(data)


@app.post("/predict/nifti", response_model=PredictResponse)
async def predict_nifti_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a NIfTI file → return segmentation mask as NIfTI."""
    # Save uploaded bytes to a real temporary file for nibabel
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        volume, affine = load_nifti(tmp.name)

    mask = predict(MODEL, volume)
    out_bytes = nifti_to_bytes(mask, affine)

    return StreamingResponse(
        out_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mask.nii.gz"},
    )


@app.post("/predict/dicom", response_model=PredictResponse)
async def predict_dicom_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a ZIP of DICOM slices → return segmentation mask as NIfTI."""
    volume = load_dicom_zip(file)
    mask = predict(MODEL, volume)

    affine = np.eye(4)
    out_bytes = nifti_to_bytes(mask, affine)

    return StreamingResponse(
        out_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mask.nii.gz"},
    )


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "model": "VNet-TF2", "message": "Inference API running"}
