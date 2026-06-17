# VNet‑TF2‑Inference

TensorFlow 2 / Keras re‑implementation of the V‑Net architecture for 3D medical image segmentation.  
Based on the original TensorFlow 1.x code by Miguel Monteiro:  
[https://github.com/MiguelMonteiro/VNet-Tensorflow](https://github.com/MiguelMonteiro/VNet-Tensorflow)

This version provides:

- a TF2/Keras V‑Net model  
- dynamic‑shape inference  
- NIfTI and DICOM loaders  
- a command‑line interface  
- a FastAPI inference server  

Training code is not included.

---

## Architecture

V‑Net is a 3D encoder–decoder network with residual connections.

This implementation:

- uses Keras `Layer` subclasses (`ConvBlock`, `ConvBlock2`, `VNet`)  
- creates all variables in `__init__`  
- supports `input_shape=(None, None, None, C)`  
- exports as a standard Keras model  

The diagram is included as `VNetDiagram.png`.

---

## Installation

```
git clone https://github.com/yourusername/vnet-tf2-inference
cd vnet-tf2-inference
pip install -e .
```

Optional test dependencies:

```
pip install -e .[test]
```

Python 3.10+ required.

---

## Python API

```python
from vnet.inference import load_vnet, predict_nifti

model = load_vnet(
    weights_path="weights.h5",
    input_shape=(None, None, None, 1),
    num_classes=1,
)

mask, affine = predict_nifti(model, "scan.nii.gz")
```

The output mask matches the input spatial shape.

---

## Command‑Line Interface

Installed as `vnet-infer`.

### Global options

```
--weights PATH        Path to model weights (.h5 or checkpoint)
--classes N           Number of output classes (default: 1)
--no-activation       Return raw logits
```

### NIfTI

```
vnet-infer [--weights W] [--classes N] [--no-activation] \
    nifti input.nii.gz output.nii.gz
```

### DICOM folder

```
vnet-infer [--weights W] [--classes N] [--no-activation] \
    dicom ./dicom_series/ output.nii.gz
```

Output is a NIfTI mask.

---

## FastAPI Server

Start:

```
uvicorn api.server:app --reload
```

### Endpoints

- `POST /predict/nifti` — NIfTI file  
- `POST /predict/dicom` — ZIP containing a DICOM series  

Returns HTTP 400 for invalid ZIPs or unreadable DICOM data.

---

## Model Configuration

```python
from vnet.model import build_vnet

model = build_vnet(
    input_shape=(None, None, None, 1),
    num_classes=3,
    num_channels=32,
    num_levels=4,
    num_convolutions=(1, 2, 3, 3),
    bottom_convolutions=3,
)
```

Parameters:

- `num_classes`  
- `num_channels`  
- `num_levels`  
- `num_convolutions`  
- `bottom_convolutions`  
- `activation`  
- `dropout`  

---

## Inference Utilities

The inference module provides:

- NIfTI I/O  
- DICOM loading  
- normalization  
- channel handling  
- sigmoid or softmax output  

Functions:

- `predict(model, volume, raw=False)`  
- `predict_nifti(model, path)`  
- `predict_dicom(model, folder)`  

Supports arbitrary spatial dimensions.

---

## Docker

Build:

```
docker build -t vnet-tf2 .
```

Run:

```
docker run -p 8000:8000 vnet-tf2
```

Server:

- http://localhost:8000  
- http://localhost:8000/docs  

---

## Benchmarking

```
python bench/benchmark_vnet.py --shape 64 64 64 --runs 20
```

Options:

- `--shape D H W`  
- `--runs N`  
- `--warmup N`  

Example output:

```
=== VNet Inference Benchmark ===
Volume shape: (64, 64, 64)
Runs: 20
Average latency: 12.34 ms
P50 latency:    11.90 ms
P90 latency:    13.10 ms
P99 latency:    14.02 ms
Throughput:     81.0 volumes/sec
```

---

## License

MIT license.  
This project is a clean re‑implementation.
