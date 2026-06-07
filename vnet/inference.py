from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pydicom
import tensorflow as tf

from .model import build_vnet


def load_vnet(
    weights_path: str | None = None,
    input_shape: tuple[int | None, int | None, int | None, int] = (None, None, None, 1),
    num_classes: int = 1,
) -> tf.keras.Model:
    """
    Build a VNet model and optionally load weights.
    Supports dynamic spatial dimensions.
    """
    model = build_vnet(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D volume to float32 in [0, 1].
    Ensures shape (D, H, W, C).
    """
    volume = volume.astype(np.float32)

    # Add channel dimension if grayscale
    if volume.ndim == 3:
        volume = volume[..., np.newaxis]

    # Normalize to [0, 1]
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)

    return volume


def predict(
    model: tf.keras.Model,
    volume: np.ndarray,
    apply_activation: bool = True,
) -> np.ndarray:
    """
    Run inference on a preprocessed volume.
    Returns segmentation mask with same spatial shape.
    """
    x = preprocess_volume(volume)
    x = np.expand_dims(x, axis=0)  # add batch dimension

    logits = model.predict(x, verbose=0)[0]

    if not apply_activation:
        return logits

    # Binary segmentation
    if logits.shape[-1] == 1:
        return tf.nn.sigmoid(logits).numpy()

    # Multi-class segmentation
    return tf.nn.softmax(logits, axis=-1).numpy()


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return (volume, affine).
    """
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine


def predict_nifti(
    model: tf.keras.Model,
    path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI → preprocess → predict → return (mask, affine).
    """
    volume, affine = load_nifti(path)
    mask = predict(model, volume)
    return mask, affine


def save_nifti(mask: np.ndarray, affine: np.ndarray, out_path: str) -> None:
    """
    Save a predicted mask as NIfTI.
    """
    nii = nib.Nifti1Image(mask.astype(np.float32), affine)
    nib.save(nii, out_path)


def load_dicom_series(folder: str) -> np.ndarray:
    """
    Load a DICOM series from a folder into a 3D numpy array.
    Assumes one slice per file.
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ]

    datasets = [pydicom.dcmread(f) for f in files]

    # Sort slices by Z position
    datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    slices = [d.pixel_array for d in datasets]
    volume = np.stack(slices, axis=0).astype(np.float32)

    return volume


def predict_dicom(
    model: tf.keras.Model,
    folder: str,
) -> np.ndarray:
    """
    Load DICOM folder → preprocess → predict → return mask.
    """
    volume = load_dicom_series(folder)
    return predict(model, volume)
