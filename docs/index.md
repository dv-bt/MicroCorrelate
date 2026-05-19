# MicroCorrelate

A Python library for correlative microscopy workflows: stitch multi-tile
acquisitions from Thermo Fisher Maps and register images across modalities.

```{toctree}
:maxdepth: 1
:hidden:

examples
api
```

## Overview

MicroCorrelate covers three areas:

- **Stitching**: assemble tiled datasets from the Thermo Fisher Maps software
  into TIFF or OME-Zarr files, with pixel size and stage position preserved in
  the metadata.
- **Registration**: align images from different modalities onto a common
  coordinate system via intensity-based registration (SimpleITK) or interactive
  landmark placement (Napari).
- **I/O**: read common correlative microscopy formats including TIFF, HDF5
  (LA-ICP-MS), OME-Zarr, and ToF-SIMS text files.

## Installation

```bash
pip install microcorrelate
```

See {doc}`examples` for usage.