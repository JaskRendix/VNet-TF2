from __future__ import annotations

from pathlib import Path
from typing import Literal

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
        # For inference-only loading, we don't need optimizer state.
        model.load_weights(weights_path).expect_partial()

    return model


def preprocess_volume(
    volume: np.ndarray,
    add_channel: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Normalize a 3D volume to float32 in [0, 1] and ensure shape (D, H, W, C).

    Parameters
    ----------
    volume:
        Input volume as a NumPy array. Expected shape (D, H, W) or (D, H, W, C).
    add_channel:
        If True, add a singleton channel dimension when volume.ndim == 3.
    normalize:
        If True, normalize intensities to [0, 1] using min-max scaling.
    """
    volume = volume.astype(np.float32)

    if add_channel and volume.ndim == 3:
        volume = volume[..., np.newaxis]

    if normalize:
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = np.zeros_like(volume)

    return volume


def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid in NumPy."""
    out = np.empty_like(x, dtype=np.float32)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)

    return out


def _softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in NumPy."""
    shift = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shift)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def predict(
    model: tf.keras.Model,
    volume: np.ndarray,
    apply_activation: bool = True,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Run inference on a volume.

    Parameters
    ----------
    model:
        Keras model returned by `load_vnet` or `build_vnet`.
    volume:
        Input volume as a NumPy array, shape (D, H, W) or (D, H, W, C).
    apply_activation:
        If True, apply sigmoid (binary) or softmax (multi-class) to logits.
    preprocess:
        If True, run `preprocess_volume` before inference.

    Returns
    -------
    mask:
        Segmentation output with same spatial shape as input volume.
    """
    x = preprocess_volume(volume) if preprocess else volume
    x = np.expand_dims(x, axis=0)  # (1, D, H, W, C)

    logits = model.predict(x, verbose=0)[0]

    if not apply_activation:
        return logits

    if logits.shape[-1] == 1:
        return _sigmoid_numpy(logits)

    return _softmax_numpy(logits, axis=-1)


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file and return (volume, affine).

    Returns
    -------
    volume:
        3D or 4D NumPy array (D, H, W) or (D, H, W, C) in float32.
    affine:
        4x4 affine matrix from the NIfTI header.
    """
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine


def predict_nifti(
    model: tf.keras.Model,
    path: str,
    apply_activation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI → preprocess → predict → return (mask, affine).
    """
    volume, affine = load_nifti(path)
    mask = predict(model, volume, apply_activation=apply_activation)
    return mask, affine


def save_nifti(mask: np.ndarray, affine: np.ndarray, out_path: str) -> None:
    """
    Save a predicted mask as NIfTI.

    If mask is float32, it is saved as float32; otherwise uint8 is used.
    """
    dtype = np.float32 if mask.dtype == np.float32 else np.uint8
    nii = nib.Nifti1Image(mask.astype(dtype), affine)
    nib.save(nii, out_path)


def load_dicom_series(folder: str) -> np.ndarray:
    """
    Load a DICOM series from a folder into a 3D NumPy array.

    - Supports both single-slice-per-file series and multi-frame DICOMs.
    - Uses a stable sort based on ImagePositionPatient, InstanceNumber, and filename.

    Returns
    -------
    volume:
        3D NumPy array with shape (D, H, W) in float32.
    """
    folder = Path(folder)

    # Collect DICOM-like files
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".dcm", ".ima"}]

    if not files:
        raise FileNotFoundError(f"No valid DICOM files found in directory: {folder}")

    # Read datasets
    file_dataset_pairs = [(p, pydicom.dcmread(p)) for p in files]

    # Multi-frame DICOM (NumberOfFrames)
    first_ds = file_dataset_pairs[0][1]
    if hasattr(first_ds, "NumberOfFrames"):
        return first_ds.pixel_array.astype(np.float32)

    # Stable sorting key
    def sort_key(pair):
        path, ds = pair

        # Primary: ImagePositionPatient (z-axis)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) >= 3:
            try:
                return (float(ipp[2]), path.name)
            except Exception:
                pass

        # Secondary: InstanceNumber
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return (int(inst), path.name)
            except Exception:
                pass

        # Fallback: filename
        return (0, path.name)

    # Sort slices deterministically
    file_dataset_pairs.sort(key=sort_key)

    # Stack slices
    slices = [ds.pixel_array for _, ds in file_dataset_pairs]
    volume = np.stack(slices, axis=0).astype(np.float32)

    return volume


def predict_dicom(
    model: tf.keras.Model,
    folder: str,
    apply_activation: bool = True,
) -> np.ndarray:
    """
    Load DICOM folder → preprocess → predict → return mask.
    """
    volume = load_dicom_series(folder)
    return predict(model, volume, apply_activation=apply_activation)


OrientationMode = Literal["none", "nifti_affine_only"]


def canonicalize_nifti_orientation(
    volume: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for orientation normalization.

    Currently returns inputs unchanged, but this is where you would:
    - reorient to a canonical space (e.g., RAS)
    - adjust the affine accordingly

    Kept as a separate function so it can be easily extended later.
    """
    return volume, affine


def patchwise_inference(
    model: tf.keras.Model,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: tuple[int, int, int] = (0, 0, 0),
    apply_activation: bool = True,
    preprocess: bool = True,
) -> np.ndarray:
    """
    Placeholder for sliding-window / patch-based inference.

    For now, this simply calls `predict` on the full volume.
    Implement patch extraction, model calls per patch, and blending
    if you need to handle very large volumes that don't fit in memory.
    """
    return predict(
        model, volume, apply_activation=apply_activation, preprocess=preprocess
    )
