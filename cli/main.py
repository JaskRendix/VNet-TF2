from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vnet.inference import load_dicom_series, load_nifti, load_vnet, predict, save_nifti


def infer_nifti(model, input_path: str | Path, output_path: str | Path) -> None:
    """Run inference on a NIfTI file and save the predicted mask."""
    volume, affine = load_nifti(str(input_path))
    mask = predict(model, volume)
    save_nifti(mask, affine, str(output_path))
    print(f"[OK] Saved mask to {output_path}")


def infer_dicom(model, folder: str | Path, output_path: str | Path) -> None:
    """Run inference on a folder containing a DICOM series."""
    volume = load_dicom_series(str(folder))
    mask = predict(model, volume)
    affine = np.eye(4)
    save_nifti(mask, affine, str(output_path))
    print(f"[OK] Saved mask to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vnet-infer",
        description="VNet-TF2 Inference CLI",
    )

    sub = parser.add_subparsers(dest="command")

    # NIfTI inference
    nifti_cmd = sub.add_parser("nifti", help="Run inference on a NIfTI volume")
    nifti_cmd.add_argument("input", type=str, help="Input NIfTI file")
    nifti_cmd.add_argument("output", type=str, help="Output NIfTI mask")

    # DICOM inference
    dicom_cmd = sub.add_parser("dicom", help="Run inference on a DICOM folder")
    dicom_cmd.add_argument("folder", type=str, help="Folder containing DICOM slices")
    dicom_cmd.add_argument("output", type=str, help="Output NIfTI mask")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load model once — dynamic spatial dims
    model = load_vnet(
        weights_path=None,
        input_shape=(None, None, None, 1),
        num_classes=1,
    )

    if args.command == "nifti":
        infer_nifti(model, args.input, args.output)

    elif args.command == "dicom":
        infer_dicom(model, args.folder, args.output)

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
