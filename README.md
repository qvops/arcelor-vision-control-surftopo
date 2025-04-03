# Surftopo

**Surftopo** is a real-time system for advanced height map processing.

## Installation

```sh
pip install requirements.txt
```

## Usage

```python
import surftopo.topography

options = surftopo.topography.Options()
results = surftopo.topography.analyze_surface_topography(x, y, z, options)
```