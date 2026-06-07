## VNet‑TF2‑Inference

TensorFlow 2 / Keras re‑implementation of the V‑Net architecture for 3D medical image segmentation.  
This project provides a modernized, inference‑focused rewrite of the original TensorFlow 1.x implementation by Miguel Monteiro:

Original repository:  
`https://github.com/MiguelMonteiro/VNet-Tensorflow`

This version includes:

- a fully TF2/Keras‑compliant V‑Net model  
- dynamic‑shape inference (any 3D volume size)  
- NIfTI and DICOM loaders  
- a command‑line interface  
- a FastAPI inference server  

Training code is intentionally not included — this project is optimized for **deployment‑ready inference**.

---

## Architecture

V‑Net is a fully convolutional 3D encoder–decoder network with residual connections, following the original paper:

* V‑Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation  
  [https://arxiv.org/abs/1606.04797](https://arxiv.org/abs/1606.04797)

This implementation:

- uses proper Keras `Layer` subclasses (`ConvBlock`, `ConvBlock2`, `VNet`)  
- creates all variables in `__init__` (no TF1‑style graph ops)  
- supports arbitrary spatial dimensions via `input_shape=(None, None, None, C)`  
- is compatible with `model.predict()` and SavedModel export  

The diagram from the original repository is included as `VNetDiagram.png`.

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

Python 3.10+ is required.

---

## Python API

Example:

```python
from vnet.inference import load_vnet, predict_nifti

model = load_vnet(
    weights_path="weights.h5",
    input_shape=(None, None, None, 1),  # dynamic spatial dims
    num_classes=1,
)

mask, affine = predict_nifti(model, "scan.nii.gz")
```

`mask` is a NumPy array with the same spatial dimensions as the input volume.

---

## Command‑Line Interface

The CLI is installed as `vnet-infer`.

### NIfTI

```
vnet-infer nifti input.nii.gz output.nii.gz
```

### DICOM folder

```
vnet-infer dicom ./dicom_series/ output.nii.gz
```

The output is a NIfTI mask.

---

## FastAPI Server

Start the server:

```
uvicorn api.server:app --reload
```

### Predict from NIfTI

```
POST /predict/nifti
file=@scan.nii.gz
```

### Predict from DICOM ZIP

```
POST /predict/dicom
file=@dicom_series.zip
```

The response is a NIfTI file containing the predicted mask.

---

## Model Configuration

The model can be configured through the builder:

```python
from vnet.model import build_vnet

model = build_vnet(
    input_shape=(None, None, None, 1),  # dynamic spatial dims
    num_classes=3,
    num_channels=32,
    num_levels=4,
    num_convolutions=(1, 2, 3, 3),
    bottom_convolutions=3,
)
```

Parameters:

- **num_classes** — number of output channels  
- **num_channels** — base number of filters  
- **num_levels** — number of encoder/decoder levels  
- **num_convolutions** — convolutions per level  
- **bottom_convolutions** — convolutions at the bottom level  
- **activation** — activation function  
- **dropout** — dropout rate  

---

## Inference Details

The inference module provides:

- NIfTI loading and saving  
- DICOM series loading  
- preprocessing (normalization, channel handling)  
- postprocessing (sigmoid or softmax)  

Functions:

- `predict(model, volume)`  
- `predict_nifti(model, path)`  
- `predict_dicom(model, folder)`  

All inference functions support arbitrary spatial dimensions.

---

Giorgio, your README is already clean and professional — it just needs one more section to document your new benchmarking tool.  
Here’s the **updated README**, with a polished, production‑grade **Benchmarking** section added in the right place and tone.

I’ll give you only the updated portion so you can drop it in without rewriting the whole file.

---

## Benchmarking

This project includes a lightweight benchmarking tool to measure V‑Net inference performance on arbitrary 3D volumes.

Run the benchmark:

```
python bench/benchmark_vnet.py --shape 64 64 64 --runs 20
```

Options:

- `--shape D H W` — volume size (default: `64 64 64`)
- `--runs N` — number of timed inference runs (default: `10`)
- `--warmup N` — warmup iterations before timing (default: `3`)

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

The benchmark uses the same inference pipeline as the API and CLI, ensuring results reflect real‑world performance.

---

## Original Repository

This project is a TF2/Keras rewrite of:

`https://github.com/MiguelMonteiro/VNet-Tensorflow`

The original code is TensorFlow 1.x and implements only the computation graph.  
This project replaces the TF1 graph with a TF2/Keras model and adds inference tooling.

---

## License

The original implementation is MIT‑licensed.  
This project is a clean re‑implementation and does not reuse TensorFlow 1.x code.
