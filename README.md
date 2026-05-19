# MicroCorrelate

[![PyPI version](https://img.shields.io/pypi/v/microcorrelate)](https://pypi.org/project/microcorrelate/)
[![Python versions](https://img.shields.io/pypi/pyversions/microcorrelate)](https://pypi.org/project/microcorrelate/)
[![CI](https://github.com/dv-bt/MicroCorrelate/actions/workflows/ci.yml/badge.svg)](https://github.com/dv-bt/MicroCorrelate/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/github/license/dv-bt/MicroCorrelate)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://dv-bt.github.io/MicroCorrelate/)

A simple Python library for handling correlative microscopy datasets.

Visit the [MicroCorrelate website](https://dv-bt.github.io/MicroCorrelate/) for the full documentation.

## Features

- **Stitching**: assemble tiled datasets from [Thermo Fisher Maps](https://www.thermofisher.com/ch/en/home/electron-microscopy/products/software-em-3d-vis/maps-software.html) into single TIFF or OME-Zarr files, with physical pixel size and stage position metadata
- **Intensity-based registration**: automatic multi-modal image registration via [SimpleITK](https://simpleitk.org/) (affine, rigid, similarity transforms)
- **Landmark-based registration**: interactive registration using a [Napari](https://napari.org/) viewer for manual point placement

## Installation

We recommend installing the package in a fresh environment:

```bash
pip install microcorrelate
```

## Command-line interface

A `stitch-maps` command is available after installation for stitching tiled
datasets from Thermo Fisher Maps:

```bash
stitch-maps --source data/LayersData/tileset --dest out/stitched.zarr --verbose
```

Run `stitch-maps --help` for the full list of options.