# Surftopo

**Surftopo** is a real-time system for advanced height map processing.

## Installation

```sh
pip install requirements.txt
```

## Usage

```python
import  surftopo.topography

options = surftopo.topography.Options()
results = surftopo.topography.analyze_surface_topography(x, y, z, options)
```
## Tensor device selection

Surftopo can be configured to select in which device will the tensor operations run on.

```python

# CPU

    opt_object = Options()
    opt_object.device = "cpu"
    opt_object.polynomial_device = "cpu"

# CUDA, needs a Nvidia gpu

    opt_object = Options()
    opt_object.device = "cuda"
    opt_object.polynomial_device = "cuda"

# METAL, needs an Apple Silicon processor.
# Note: not all Pytorch functions are available with mps

    opt_object = Options()
    opt_object.device = "mps"
    opt_object.polynomial_device = "mps"
```
